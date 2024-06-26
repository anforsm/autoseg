import os
import random
from pathlib import Path
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
from einops import rearrange

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import zarr

from autoseg.models import ExampleModel, ExampleModel2D, ConfigurableUNet, Model
import autoseg.losses as losses
from autoseg.datasets import GunpowderZarrDataset
from autoseg.config import read_config
from autoseg.datasets.utils import multisample_collate as collate
from autoseg.transforms.gp_parser import snake_case_to_camel_case
from autoseg.log import Logger
from autoseg.train_utils import get_2D_snapshot, save_zarr_snapshot
from autoseg.utils import get_artifact_base_path
import autoseg.optimizers as optim

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

# CONFIG_PATH = "defaults"
CONFIG_PATH = "examples/lsd"
# CONFIG_PATH = "autoseg/user_configs/test/config"

WORLD_SIZE = torch.cuda.device_count()
DEVICE = 0


def ddp_setup(rank: int, world_size: int):
    """
    Args:
      rank: Unique identifier of each process
      world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12313"
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def batch_predict(
    model,
    batch,
    model_inputs,
    model_outputs,
    batch_outputs,
    crit=None,
    loss_inputs=None,
):
    """Predict on a batch and calculate loss if crit is provided.

    Args:
        model: torch.nn.Module
        batch: tuple
        crit: torch.nn.Module, optional

    Returns:
        prediction: torch.Tensor
        loss: torch.Tensor
    """
    # raw, labels, affs, affs_weights = batch

    batch = [
        torch.tensor(x).to(torch.float32).to(DEVICE) if not x.dtype == np.uint64 else x
        for x in batch
    ]
    batch_outputs = {
        output_name: output for output_name, output in zip(batch_outputs, batch)
    }

    # raw: (B, C, Z, Y, X)

    # in case of multiple inputs to the model
    # we concatenate them along the channel dimension
    inp = torch.cat(tuple(batch_outputs[name] for name in model_inputs), dim=1)
    prediction = model(inp)

    if not isinstance(prediction, tuple):
        prediction = [prediction]

    model_outputs = {
        output_name: output for output_name, output in zip(model_outputs, prediction)
    }

    if crit is not None and loss_inputs is not None:
        vars_ = {**model_outputs, **batch_outputs}
        loss = crit(*[vars_[name] for name in loss_inputs])
        return model_outputs, loss

    return model_outputs


def save_model(model, **kwargs):
    if MULTI_GPU:
        if DEVICE == "cuda:0":
            model.module.save(**kwargs)
    else:
        model.save(**kwargs)


def train(
    model,
    dataloader,
    crit,
    optimizer,
    batch_outputs,
    model_inputs,
    model_outputs,
    loss_inputs,
    logger=None,
    val_dataloader=None,
    # learning_rate=1e-5,
    update_steps=10000,
    log_snapshot_every=10,
    save_every=1000,
    val_log=10_000,
    overwrite_checkpoints=True,
    save_best=True,
    snapshot_dir="",
):
    master_process = not MULTI_GPU or DEVICE == "cuda:0"
    # crit = torch.nn.MSELoss()
    # optimizer = optimizer(model.parameters(), lr=learning_rate)

    avg_loss = 0
    lowest_val_loss = float("inf")

    for step, batch in zip(range(update_steps + 1), iter(dataloader)):
        optimizer.zero_grad()

        prediction, loss = batch_predict(
            model, batch, model_inputs, model_outputs, batch_outputs, crit, loss_inputs
        )
        loss.backward()
        optimizer.step()

        # Log training loss in console
        if not MULTI_GPU or DEVICE == "cuda:0":
            print(
                f"Step {step}/{update_steps}, loss: {loss.item():.4f}, val: {avg_loss:.4f}",
                end="\r",
            )

        # Log training loss n wandb

        if not logger is None and master_process:
            logger.push(
                {
                    "step": step,
                    "loss": loss.item(),
                    "num_images": step,  # * 12,
                    "epoch": step / 544,  # should divide by batch size as well
                }
            )

        if step % log_snapshot_every == 0 and master_process:
            image_tensors = {}
            source_dict = prediction | {
                name: val for name, val in zip(batch_outputs, batch)
            }

            for name in logger.image_keys:
                image_tensors[name] = source_dict[name]

            images = get_2D_snapshot(image_tensors, center_crop=False)
            logger.push({"images": list(images)})

            zarrs = save_zarr_snapshot(
                (Path(snapshot_dir) / Path("snapshots.zarr")).as_posix(),
                f"{step}",
                image_tensors,
            )
            logger.push({"zarrs": zarrs})

        # Save model
        if step % save_every == 0 and master_process and not step == 0:
            save_model(model, step=step, overwrite_checkpoints=overwrite_checkpoints)

        # Log validation and snapshots
        if (
            step % val_log == 0
            and val_dataloader is not None
            and master_process
            and not step == 0
        ):
            with torch.no_grad():
                model.eval()
                avg_loss = 0
                num_val_batches = 10
                for i, batch in zip(range(num_val_batches), iter(val_dataloader)):
                    prediction, loss = batch_predict(
                        model,
                        batch,
                        model_inputs,
                        model_outputs,
                        batch_outputs,
                        crit,
                        loss_inputs,
                    )
                    avg_loss += loss.item()

                avg_loss /= num_val_batches

                if avg_loss < lowest_val_loss and save_best:
                    lowest_val_loss = avg_loss
                    save_model(model, save_best=save_best)

                if not logger is None and master_process:
                    logger.push({"val_loss": avg_loss})

                model.train()

        if not logger is None and master_process:
            logger.log()


def dataloader_from_config(dataset, config):
    if config["parallel"]:
        if config["use_gunpowder_precache"]:
            import gunpowder as gp

            dataset.pipeline += gp.PreCache(
                num_workers=config["num_workers"],
                cache_size=config["precache_per_worker"] * config["num_workers"],
            )
            return DataLoader(
                dataset=dataset,
                collate_fn=collate,
                pin_memory=False,
            )

        return DataLoader(
            dataset=dataset,
            collate_fn=collate,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            prefetch_factor=config["precache_per_worker"],
            pin_memory=False,
        )
    else:
        return DataLoader(
            dataset=dataset,
            collate_fn=collate,
            batch_size=config["batch_size"],
            pin_memory=False,
        )


def crit_from_config(config):
    loss_name = list(filter(lambda x: not x.startswith("_"), config.keys()))[0]
    loss_name_cls = snake_case_to_camel_case(loss_name)
    if hasattr(losses, loss_name_cls):
        crit = getattr(losses, loss_name_cls)
    else:
        crit = getattr(nn, loss_name)

    crit = crit(**config[loss_name])
    return crit


def logger_from_config(config):
    root_config = config
    config = config["training"]
    providers = []
    if "wandb" in config["logging"] and config["logging"]["wandb"]:
        providers.append("wandb")

    if "tensorboard" in config["logging"] and config["logging"]["tensorboard"]:
        providers.append("tensorboard")

    logger = Logger(provider=providers, config=root_config)
    logger.image_keys = config["logging"]["log_images"]
    return logger


def main(rank, config):
    global DEVICE, WANDB_LOG, MULTI_GPU

    if MULTI_GPU:
        ddp_setup(rank=rank, world_size=WORLD_SIZE)

    DEVICE = rank
    DEVICE = f"cuda:{DEVICE}"

    if MULTI_GPU:
        WANDB_LOG = WANDB_LOG and DEVICE == 0

    # if WANDB_LOG:
    #    wandb.init(project="autoseg")

    model = Model(config)
    model = model.to(DEVICE)

    if MULTI_GPU:
        model = DDP(model, device_ids=[DEVICE])

    crit = crit_from_config(config["training"]["loss"])

    dataset = GunpowderZarrDataset(
        config=config["pipeline"],
        input_image_shape=config["model"]["input_image_shape"],
        output_image_shape=config["model"]["output_image_shape"],
    )

    dataloader = dataloader_from_config(
        dataset=dataset, config=config["training"]["train_dataloader"]
    )

    if "val_dataloader" in config["training"] and config["training"]["val_dataloader"]:
        validation_dataset = GunpowderZarrDataset(
            config=config["training"]["val_dataloader"]["pipeline"],
            input_image_shape=config["model"]["input_image_shape"],
            output_image_shape=config["model"]["output_image_shape"],
        )

        val_dataloader = dataloader_from_config(
            dataset=validation_dataset, config=config["training"]["val_dataloader"]
        )
    else:
        val_dataloader = None

    root_config = config
    base_path = get_artifact_base_path(config)
    config = config["training"]
    batch_outputs = config["batch_outputs"]
    model_outputs = config["model_outputs"]
    model_inputs = config["model_inputs"]
    loss_inputs = config["loss"]["_inputs"]

    optimizer = (
        list(config["optimizer"].keys())[0] if "optimizer" in config else "AdamW"
    )
    kwargs = config["optimizer"][optimizer]
    if hasattr(optim, optimizer):
        optimizer = getattr(optim, optimizer)
    elif hasattr(torch.optim, optimizer):
        optimizer = getattr(torch.optim, optimizer)
    optimizer = optimizer(model.parameters(), **kwargs)

    train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        crit=crit,
        optimizer=optimizer,
        batch_outputs=batch_outputs,
        model_outputs=model_outputs,
        model_inputs=model_inputs,
        loss_inputs=loss_inputs,
        logger=logger_from_config(root_config)
        if not MULTI_GPU or DEVICE == "cuda:0"
        else None,
        log_snapshot_every=config["log_snapshot_every"],
        save_every=config["save_every"],
        val_log=config["val_log"],
        overwrite_checkpoints=config["overwrite_checkpoints"],
        save_best=config["save_best"],
        update_steps=config["update_steps"],
        snapshot_dir=base_path + "snapshots",
    )

    if MULTI_GPU:
        destroy_process_group()


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = CONFIG_PATH

    config = read_config(config_path)
    import json

    with open(get_artifact_base_path(config) + "config.json", "w") as f:
        json.dump(config, f, indent=4)
    MULTI_GPU = config["training"]["multi_gpu"]
    WANDB_LOG = config["training"]["logging"]["wandb"]

    if WANDB_LOG:
        import wandb

    if MULTI_GPU:
        mp.spawn(main, args=(config,), nprocs=WORLD_SIZE)
    else:
        main(rank=0, config=config)
