import os

import torch
from torch.utils.data import DataLoader

import numpy as np
from einops import rearrange

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import ExampleModel
from losses import WeightedMSELoss
from datasets import GunpowderZarrDataset, Kh2015
from config import read_config

pipeline = None

DEVICE = "cuda"

WANDB_LOG = False
if WANDB_LOG:
    import wandb


def train(
    model,
    dataset,
    val_dataset=None,
    batch_size=1,
    learning_rate=1e-5,
    update_steps=1000,
):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=5, prefetch_factor=4
    )

    crit = WeightedMSELoss()
    # crit = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step = 0

    if val_dataset is not None:
        val_iter = iter(
            DataLoader(
                val_dataset, batch_size=batch_size, num_workers=5, prefetch_factor=4
            )
        )

    val_log = 1000

    avg_loss = 0
    lowest_val_loss = float("inf")

    batch_iterator = iter(dataloader)
    for raw, labels, affs, affs_weights in batch_iterator:
        raw = torch.tensor(raw.copy()).to(DEVICE)
        # raw = raw[None, None, ...]
        # raw: (B, C, Z, Y, X)
        affs = torch.tensor(affs.copy()).to(torch.float32).to(DEVICE)
        # affs = affs[None, ...]
        affs_weights = torch.tensor(affs_weights.copy()).to(torch.float32).to(DEVICE)
        # affs_weights = affs_weights[None, ...]

        optimizer.zero_grad()
        prediction = model(raw)
        loss = crit(prediction, affs, affs_weights)
        loss.backward()
        optimizer.step()
        step += 1

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
            torch.save(model.state_dict(), "out/latest_model3.pt")
            with torch.no_grad():
                model.eval()
                raw, labels, affs, affs_weights = next(batch_iterator)

                raw = torch.tensor(raw.copy()).to(DEVICE)
                raw = raw[0].unsqueeze(0)
                affs = torch.tensor(affs.copy()).to(torch.float32).to(DEVICE)
                affs = affs[0].unsqueeze(0)
                affs_weights = (
                    torch.tensor(affs_weights.copy()).to(torch.float32).to(DEVICE)
                )
                affs_weights = affs_weights[0].unsqueeze(0)

                prediction = model(raw)
                loss = crit(prediction, affs, affs_weights)

                z_raw = raw.shape[-3] // 2
                z_label = labels.shape[-3] // 2
                raw = raw.squeeze()[z_raw, :, :].unsqueeze(0).cpu().numpy()
                prediction = prediction.squeeze()[:, z_label, :, :].cpu().numpy()
                affs = affs.squeeze()[:, z_label, :, :].cpu().numpy()
                labels = labels[z_label, :, :]

                raw = rearrange(raw, "c h w -> h w c")
                prediction = rearrange(prediction, "c h w -> h w c")
                affs = rearrange(affs, "c h w -> h w c")

                wandb.log(
                    {
                        "step": step,
                        "raw": wandb.Image(
                            raw,
                            # masks={
                            #  "labels": {"mask_data": labels}
                            # }
                        ),
                        "predicted affs": wandb.Image(
                            prediction,
                            # masks={
                            #  "labels": {"mask_data": labels}
                            # }
                        ),
                        "affs": wandb.Image(
                            affs,
                            # masks={
                            #  "labels": {"mask_data": labels}
                            # }
                        ),
                    }
                )

                if val_dataset is not None:
                    avg_loss = 0
                    num_val_batches = 10
                    for _ in range(num_val_batches):
                        raw, labels, affs, affs_weights = next(val_iter)
                        raw = torch.tensor(raw.copy()).to(DEVICE)
                        affs = torch.tensor(affs.copy()).to(torch.float32).to(DEVICE)
                        affs_weights = (
                            torch.tensor(affs_weights.copy())
                            .to(torch.float32)
                            .to(DEVICE)
                        )

                        prediction = model(raw)
                        loss = crit(prediction, affs, affs_weights)
                        avg_loss += loss.item()

                    avg_loss /= num_val_batches
                    if avg_loss < lowest_val_loss:
                        lowest_val_loss = avg_loss
                        torch.save(model.state_dict(), "out/best_model3.pt")
                    wandb.log({"val_loss": avg_loss})

                model.train()

        if step >= update_steps:
            break


if __name__ == "__main__":
    if WANDB_LOG:
        wandb.init(project="autoseg")
    model = ExampleModel()
    model = torch.compile(model)
    model.to(DEVICE)

    dataset = Kh2015(
        transform=read_config("examples/no_augments")["pipeline"],
        input_shape=(36, 212, 212),
        output_shape=(12, 120, 120),
    )

    train(model, dataset)
