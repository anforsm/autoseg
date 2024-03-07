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

from autoseg.models import ExampleModel
from autoseg.losses import WeightedMSELoss
from autoseg.datasets import GunpowderZarrDataset, Kh2015
from autoseg.config import read_config
from autoseg.datasets.utils import multisample_collate as collate
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

pipeline = None

WORLD_SIZE = torch.cuda.device_count()
DEVICE = 0

WANDB_LOG = False
if WANDB_LOG:
    import wandb

MULTI_GPU = True

def ddp_setup(rank: int, world_size: int):
    """
    Args:
      rank: Unique identifier of each process
      world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12310"
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_2D_snapshot(raw, labels, affs, prediction):
    # Get the middle slice of the 3D volume
    # raw: (B, C, Z, Y, X)
    z_raw_i = raw.shape[-3] // 2
    # labels: (B, C, Z, Y, X)
    z_label_i = labels.shape[-3] // 2

    raw = raw[0, :, z_raw_i, :, :].cpu().numpy()
    prediction = prediction[0, :, z_label_i, :, :].cpu().numpy()
    affs = affs[0, :, z_label_i, :, :].cpu().numpy()
    labels = labels[z_label_i, :, :]

    raw = rearrange(raw, "c h w -> h w c")
    prediction = rearrange(prediction, "c h w -> h w c")
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
    raw, labels, affs, affs_weights = batch
    raw = torch.tensor(raw.copy()).to(DEVICE)
    # raw: (B, C, Z, Y, X)
    affs = torch.tensor(affs.copy()).to(torch.float32).to(DEVICE)
    affs_weights = torch.tensor(affs_weights.copy()).to(torch.float32).to(DEVICE)

    prediction = model(raw)
    if crit is not None:
        loss = crit(prediction, affs, affs_weights)
        print(loss)
        return prediction, loss

    return prediction


def train(
    model,
    dataset,
    val_dataset=None,
    batch_size=1,
    learning_rate=1e-5,
    update_steps=1000,
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=6,
        prefetch_factor=5,
        collate_fn=collate,
        pin_memory=True,
        # sampler=DistributedSampler(dataset) if MULTI_GPU else None,
    )

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

    val_log = 1000

    avg_loss = 0
    lowest_val_loss = float("inf")

    print("Starting training")
    batch_iterator = iter(dataloader)
    for batch in batch_iterator:
        print("loop")
        optimizer.zero_grad()

        _, loss = batch_predict(model, batch, crit)
        loss.backward()
        optimizer.step()
        step += 1

        print(
            f"Step {step}/{update_steps}, loss: {loss.item():.4f}, val: {avg_loss:.4f}",
            # end="\r",
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
                loss, prediction = batch_predict(model, batch, crit)
                raw_image, labels_image, affs_image, prediction_image = get_2D_snapshot(
                    raw, labels, affs, prediction
                )

                wandb.log(
                    {
                        "step": step,
                        "snapshots": [raw_image, affs_image, prediction_image],
                    }
                )

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


def main(rank):
    if MULTI_GPU:
        ddp_setup(rank=rank, world_size=WORLD_SIZE)
    global DEVICE, WANDB_LOG
    DEVICE = rank
    DEVICE = f"cuda:{DEVICE}"
    print("Using device", DEVICE)

    if MULTI_GPU:
        WANDB_LOG = WANDB_LOG and DEVICE == 0

    if WANDB_LOG:
        wandb.init(project="autoseg")
    print("Instatiating model")
    model = ExampleModel()
    # model = torch.nn.Linear(1, 1)
    # model = torch.compile(model)
    print("Moving model to", DEVICE)
    model = model.to(DEVICE)
    print("Wrapping model in DDP")
    if MULTI_GPU:
        model = DDP(model, device_ids=[DEVICE])
    print("Moved model to device")

    dataset = Kh2015(
        #transform=read_config("examples/no_augments")["pipeline"],
        transform=read_config("defaults")["pipeline"],
        input_shape=(36, 212, 212),
        output_shape=(12, 120, 120),
    )
    print("Loaded dataset")

    train(model, dataset, batch_size=8)

    if MULTI_GPU:
        destroy_process_group()


if __name__ == "__main__":
    if MULTI_GPU:
        mp.spawn(main, nprocs=WORLD_SIZE)
    else:
        main(rank=0)
