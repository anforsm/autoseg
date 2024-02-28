import torch
import numpy as np
from einops import rearrange

from models import ExampleModel
from losses import WeightedMSELoss
from datasets import ZarrDataset, load_dataset

WANDB_LOG = True
if WANDB_LOG:
    import wandb
pipeline = None

DEVICE = "cuda"


def train(model, dataset, val_dataset=None):
    batch_size = 1
    crit = WeightedMSELoss()
    # crit = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    steps = 20_000
    step = 0
    input_size = (36, 212, 212)
    output_size = (12, 120, 120)

    if val_dataset is not None:
        val_iter = iter(val_dataset.request_batch(input_size, output_size))

    val_log = 100

    batch_iterator = iter(dataset.request_batch(input_size, output_size))
    avg_loss = 0
    lowest_val_loss = float("inf")
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
            f"Step {step}/{steps}, loss: {loss.item():.4f}, val: {avg_loss:.4f}",
            end="\r",
        )
        if WANDB_LOG:
            wandb.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "num_images": step * 12,
                    "epoch": step / 544,
                }
            )

        if step % val_log == 0:
            torch.save(model.state_dict(), "out/latest_model.pt")
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
                        torch.save(model.state_dict(), "out/best_model.pt")
                    wandb.log({"val_loss": avg_loss})

                model.train()

        if step >= steps:
            break


if __name__ == "__main__":
    if WANDB_LOG:
        wandb.init(project="autoseg")
    model = ExampleModel()
    model.to(DEVICE)
    dataset = load_dataset("SynapseWeb/kh2015/apical")
    train(model, dataset, val_dataset=load_dataset("SynapseWeb/kh2015/oblique"))