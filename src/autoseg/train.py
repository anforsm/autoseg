import os

import torch
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
from einops import rearrange

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import zarr

from autoseg.models import ExampleModel
from autoseg.losses import WeightedMSELoss
from autoseg.datasets import GunpowderZarrDataset, Kh2015
from autoseg.config import read_config
from autoseg.datasets.utils import multisample_collate as collate

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

CONFIG_PATH = "examples/kh2015_multi"

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


def batch_predict(model, batch, crit=None):
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
    affs_weights, affs, _, labels, _, raw = batch
    raw = torch.tensor(raw.copy()).to(DEVICE)
    # raw: (B, C, Z, Y, X)
    affs = torch.tensor(affs.copy()).to(torch.float32).to(DEVICE)
    affs_weights = torch.tensor(affs_weights.copy()).to(torch.float32).to(DEVICE)

    prediction = model(raw)
    if crit is not None:
        loss = crit(prediction, affs, affs_weights)
        return prediction, loss

    return prediction


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
    val_dataset=None,
    learning_rate=1e-5,
    update_steps=1000,
):
    crit = WeightedMSELoss()
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

    val_log = 10000

    avg_loss = 0
    lowest_val_loss = float("inf")

    print("Starting training")
    batch_iterator = iter(dataloader)
    for batch in batch_iterator:
        optimizer.zero_grad()

        _, loss = batch_predict(model, batch, crit)
        loss.backward()
        optimizer.step()
        step += 1

        if not MULTI_GPU or DEVICE == "cuda:0":
            print(
                f"Step {step}/{update_steps}, loss: {loss.item():.4f}, val: {avg_loss:.4f}",
                end="\r",
            )
        if WANDB_LOG:
            wandb.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "num_images": step,  # * 12,
                    "epoch": step / 544,  # should divide by batch size as well
                }
            )

        if step % val_log == 0:
            if MULTI_GPU:
                torch.save(model.module.state_dict(), "out/latest_model3.pt")
            else:
                torch.save(model.state_dict(), "out/latest_model3.pt")
            with torch.no_grad():
                model.eval()
                batch = next(batch_iterator)
                raw, labels, affs, affs_weights = batch
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
    print(MULTI_GPU)
    if MULTI_GPU:
        ddp_setup(rank=rank, world_size=WORLD_SIZE)

    DEVICE = rank
    DEVICE = f"cuda:{DEVICE}"

    if MULTI_GPU:
        WANDB_LOG = WANDB_LOG and DEVICE == 0

    if WANDB_LOG:
        wandb.init(project="autoseg")

    model = ExampleModel()
    model = model.to(DEVICE)

    if MULTI_GPU:
        model = DDP(model, device_ids=[DEVICE])

    dataset = GunpowderZarrDataset(
        config=config["pipeline"],
        input_image_shape=(36, 212, 212),
        output_image_shape=(12, 120, 120),
    )

    dataloader = dataloader_from_config(
        dataset=dataset, config=config["training"]["train_dataloader"]
    )

    train(model, dataloader)

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
