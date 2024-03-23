import os

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

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

CONFIG_PATH = "examples/kh2015_multisource"
# CONFIG_PATH = "examples/2d_multisource"

WORLD_SIZE = torch.cuda.device_count()
DEVICE = 0


def ddp_setup(rank: int, world_size: int):
    """
    Args:
      rank: Unique identifier of each process
      world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12312"
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_2D_snapshot(raw, labels, affs, prediction):
    # Get the middle slice of the 3D volume
    # raw: (B, C, Z, Y, X)
    z_raw_i = raw.shape[-3] // 2
    # labels: (B, C, Z, Y, X)
    z_label_i = labels.shape[-3] // 2

    raw = raw[0, :, z_raw_i, :, :]
    prediction = prediction[0, :, z_label_i, :, :]
    affs = affs[0, :, z_label_i, :, :]
    labels = labels[z_label_i, :, :]

    raw = rearrange(raw, "c h w -> h (w c)")  # only 1 channel
    prediction = (rearrange(prediction, "c h w -> h w c") * 255).astype(np.uint8)
    affs = rearrange(affs, "c h w -> h w c")

    # create images
    return (
        Image.fromarray(raw),
        None,
        Image.fromarray(affs),
        Image.fromarray(prediction),
    )


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

    if not isinstance(prediction, list):
        prediction = [prediction]

    model_outputs = {
        output_name: output for output_name, output in zip(model_outputs, prediction)
    }

    if crit is not None and loss_inputs is not None:
        vars_ = {**model_outputs, **batch_outputs}
        loss = crit(*[vars_[name] for name in loss_inputs])
        return model_outputs, loss

    return model_outputs


def save_zarr_snapshot(dataset_prefix, filename, raw, labels, affs, prediction):
    f = zarr.open(filename, "a")
    num_spatial_dims = 3
    raw = raw[0]
    labels = labels[0]
    affs = affs[0]
    prediction = prediction[0]

    print(raw)
    raw += 1
    raw /= 2
    raw *= 255
    print(raw)
    print(np.max(raw), np.min(raw))
    raw = raw.astype(np.uint8)
    raw_shape = np.array(raw.shape)[1:]
    label_shape = np.array(labels.shape)
    diff = raw_shape - label_shape
    offset = diff // 2
    for name, array in zip(
        ["raw", "labels", "affs", "prediction"], [raw, labels, affs, prediction]
    ):
        f[f"{dataset_prefix}/{name}"] = array
        f[f"{dataset_prefix}/{name}"].attrs["resolution"] = [1, 1, 1]
        f[f"{dataset_prefix}/{name}"].attrs["axis_names"] = ["c", "z", "y", "x"]
        if name in ["labels", "affs", "prediction"]:
            f[f"{dataset_prefix}/{name}"].attrs["offset"] = list(offset)


def train(
    model,
    dataloader,
    crit,
    batch_outputs,
    model_inputs,
    model_outputs,
    loss_inputs,
    val_dataset=None,
    learning_rate=1e-5,
    update_steps=1000,
):
    # crit = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step = 0

    if val_dataset is not None:
        val_iter = iter(
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=5,
                prefetch_factor=4,
                collate_fn=collate,
            )
        )

    val_log = 10_000
    save_every = 10

    avg_loss = 0
    lowest_val_loss = float("inf")

    print("Starting training")
    batch_iterator = iter(dataloader)
    for batch in batch_iterator:
        optimizer.zero_grad()

        _, loss = batch_predict(
            model, batch, model_inputs, model_outputs, batch_outputs, crit, loss_inputs
        )
        loss.backward()
        optimizer.step()
        step += 1

        # Log training loss in console
        if not MULTI_GPU or DEVICE == "cuda:0":
            print(
                f"Step {step}/{update_steps}, loss: {loss.item():.4f}, val: {avg_loss:.4f}",
                end="\r",
            )

        # Log training loss in wandb
        if WANDB_LOG:
            wandb.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "num_images": step,  # * 12,
                    "epoch": step / 544,  # should divide by batch size as well
                }
            )

        # Save model
        if step % save_every == 0:
            if MULTI_GPU:
                model.save()
            else:
                model.save()

        # Log validation and snapshots
        if step % val_log == 0:
            with torch.no_grad():
                model.eval()
                batch = next(batch_iterator)
                affs_weights, affs, _, labels, _, raw = batch
                prediction, loss = batch_predict(model, batch, crit)

                raw_image, labels_image, affs_image, prediction_image = get_2D_snapshot(
                    raw, labels, affs, prediction.cpu().numpy()
                )

                save_zarr_snapshot(
                    f"{step}",
                    "out/snapshot.zarr",
                    raw,
                    labels,
                    affs,
                    prediction.cpu().numpy(),
                )

                if WANDB_LOG:
                    wandb.log(
                        {
                            "step": step,
                            "snapshots": [raw_image, affs_image, prediction_image],
                        }
                    )
                else:
                    if not raw_image.mode == "RGB":
                        raw_image = raw_image.convert("RGB")
                    if not affs_image.mode == "RGB":
                        affs_image = affs_image.convert("RGB")
                    if not prediction_image.mode == "RGB":
                        prediction_image = prediction_image.convert("RGB")
                    raw_image.save(f"out/images/raw_{step}.png")
                    affs_image.save(f"out/images/affs_{step}.png")
                    prediction_image.save(f"out/images/prediction_{step}.png")

                if val_dataset is not None:
                    avg_loss = 0
                    num_val_batches = 10
                    for _ in range(num_val_batches):
                        batch = next(val_iter)
                        _, loss = batch_predict(model, crit, batch)
                        avg_loss += loss.item()

                    avg_loss /= num_val_batches

                    if avg_loss < lowest_val_loss:
                        lowest_val_loss = avg_loss
                        if MULTI_GPU:
                            torch.save(model.module.state_dict(), "out/best_model3.pt")
                        else:
                            torch.save(model.state_dict(), "out/best_model3.pt")
                    wandb.log({"val_loss": avg_loss})

                model.train()

        # End training
        if step >= update_steps:
            break


def dataloader_from_config(dataset, config):
    if config["parallel"]:
        return DataLoader(
            dataset=dataset,
            collate_fn=collate,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            prefetch_factor=config["precache_per_worker"],
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset=dataset,
            collate_fn=collate,
            batch_size=config["batch_size"],
            pin_memory=True,
        )


def main(rank, config):
    global DEVICE, WANDB_LOG, MULTI_GPU

    loss_name = list(
        filter(lambda x: not x.startswith("_"), config["training"]["loss"].keys())
    )[0]
    loss_name_cls = snake_case_to_camel_case(loss_name)
    if hasattr(losses, loss_name_cls):
        crit = getattr(losses, loss_name_cls)
    else:
        crit = getattr(nn, loss_name)

    crit = crit(**config["training"]["loss"][loss_name])

    if MULTI_GPU:
        ddp_setup(rank=rank, world_size=WORLD_SIZE)

    DEVICE = rank
    DEVICE = f"cuda:{DEVICE}"

    if MULTI_GPU:
        WANDB_LOG = WANDB_LOG and DEVICE == 0

    if WANDB_LOG:
        wandb.init(project="autoseg")

    model = Model(config)
    model = model.to(DEVICE)

    if MULTI_GPU:
        model = DDP(model, device_ids=[DEVICE])

    dataset = GunpowderZarrDataset(
        config=config["pipeline"],
        input_image_shape=config["training"]["train_dataloader"]["input_image_shape"],
        output_image_shape=config["training"]["train_dataloader"]["output_image_shape"],
    )

    dataloader = dataloader_from_config(
        dataset=dataset, config=config["training"]["train_dataloader"]
    )

    batch_outputs = config["training"]["batch_outputs"]
    model_outputs = config["training"]["model_outputs"]
    model_inputs = config["training"]["model_inputs"]
    loss_inputs = config["training"]["loss"]["_inputs"]

    train(
        model=model,
        dataloader=dataloader,
        crit=crit,
        batch_outputs=batch_outputs,
        model_outputs=model_outputs,
        model_inputs=model_inputs,
        loss_inputs=loss_inputs,
    )

    if MULTI_GPU:
        destroy_process_group()


config = read_config(CONFIG_PATH)
MULTI_GPU = config["training"]["multi_gpu"]
WANDB_LOG = config["training"]["logging"]["wandb"]

if __name__ == "__main__":
    if WANDB_LOG:
        import wandb

    if MULTI_GPU:
        mp.spawn(main, args=(config,), nprocs=WORLD_SIZE)
    else:
        main(rank=0, config=config)
