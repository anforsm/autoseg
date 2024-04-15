import multiprocessing

multiprocessing.set_start_method("fork")

import torch

from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds
import daisy

from pathlib import Path
import daisy
import json
import logging
import os
import subprocess
import time
import gunpowder as gp
import zarr
import numpy as np
import sys

import matplotlib.pyplot as plt

from typing import List
from typing import TypedDict

from autoseg.models import ExampleModel, ExampleModel2D, Model
from autoseg.config import read_config
from autoseg.datasets.load_dataset import get_dataset_path, download_dataset
from autoseg.datasets.utils import get_voxel_size, get_shape
from autoseg.train_utils import get_2D_snapshot, save_zarr_snapshot
from autoseg.datasets import GunpowderZarrDataset
from torch.utils.data import DataLoader
from autoseg.datasets.utils import multisample_collate as collate

from PIL import Image
DEVICE = "cuda"

#config = read_config("autoseg/user_configs/anton/resolution_experiments/s0")
config = read_config("autoseg/examples/lsd")

weights = torch.load("checkpoints/UNet_extra_latest_2/step-9900/ckpt.pt")

model1 = Model(config)
model2 = Model(config)
model2.load_state_dict(weights)
#print(model1)
model1.load()
print("Params:", sum(p.numel() for p in model1.parameters()))
print("Params:", sum(p.numel() for p in model2.parameters()))

equal = True
for p1, p2 in zip(model1.parameters(), model2.parameters()):
  if not torch.all(torch.isclose(p1.data, p2.data)):
    #print(p1, p2)
    equal = False

if not equal:
  print("not Equal models")
else:
  print("Equal models")


model1 = model1.to("cuda")
model2 = model2.to("cuda")
model1.eval()
model2.eval()

chunk_request = gp.BatchRequest()
raw = gp.ArrayKey("RAW")
labels = gp.ArrayKey("LABELS")
labels_mask = gp.ArrayKey("LABELS_MASK")
size = gp.Coordinate((36, 212, 212)) * gp.Coordinate((50, 2, 2))
chunk_request.add(raw, size)
chunk_request.add(labels, size)
chunk_request.add(labels_mask, size)

#download_dataset("SynapseWeb/kh2015/oblique")
pipeline = gp.ZarrSource(
  get_dataset_path("SynapseWeb/kh2015/oblique"),
  {raw: "raw/s0", labels: "labels/s0", labels_mask: "labels_mask/s0"},
  {raw: gp.ArraySpec(interpolatable=True), labels: gp.ArraySpec(interpolatable=False), labels_mask: gp.ArraySpec(interpolatable=False)}
)
pipeline += gp.RandomLocation(
  mask= labels_mask,
  min_masked= 1
)
pipeline += gp.Normalize(raw)
pipeline += gp.IntensityScaleShift(raw, 2, -1)
pipeline += gp.Unsqueeze([raw])
pipeline += gp.Unsqueeze([raw])

with gp.build(pipeline):
  #pipeline.request_batch(chunk_request)
  sample = pipeline.request_batch(chunk_request)
raw, labels, mask = sample[raw].data, sample[labels].data, sample[labels_mask].data
print(raw.shape)

"""
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

def batch_predict(
    model,
    batch,
    model_inputs,
    model_outputs,
    batch_outputs,
    crit=None,
    loss_inputs=None,
):

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

dataset = GunpowderZarrDataset(
    config=config["pipeline"],
    input_image_shape=config["training"]["train_dataloader"]["input_image_shape"],
    output_image_shape=config["training"]["train_dataloader"]["output_image_shape"],
)
"""

#dataset.pipeline = pipeline

#dataloader = dataloader_from_config(
#    dataset=dataset, config=config["training"]["train_dataloader"]
#)

config = config["training"]
batch_outputs = config["batch_outputs"]
model_outputs = config["model_outputs"]
model_inputs = config["model_inputs"]

#aff_ = gp.ArrayKey("AFF")
#lsd_ = gp.ArrayKey("LSD")
#pipeline += gp.Torch.Predict(
#  model1,
#  inputs={"input": raw},
#  outputs={0: aff_, 1: lsd_}
#  device="cuda",
#  array_specs={
#    aff_: gp.ArraySpec
#  }
#)

#batch = next(iter(dataloader))
#for b in batch:
#  print(b.shape)

#prediction = batch_predict(
#    model1, batch, model_inputs, model_outputs, batch_outputs 
#)


#pred_affs1, _ = model1(torch.tensor(batch[0]).to("cuda"))
pred_affs1, _ = model1(torch.tensor(raw).to("cuda"))
prediction = {"affs": pred_affs1}



image_tensors = {}
source_dict = prediction | {
    name: val for name, val in zip(batch_outputs, {"raw": raw})
}

source_dict["raw"] = raw

for name in ["raw", "affs"]:
    image_tensors[name] = source_dict[name]

images = get_2D_snapshot(image_tensors)
images[0].save("1.png")
images[1].save("2.png")


zarrs = save_zarr_snapshot(
    "test.zarr",
    f"test",
    image_tensors,
)

# print(raw.shape, labels.shape)
# raw = raw.astype(np.float32)
# print(raw.min(), raw.max())
# #raw /= 255
# 
# 
# img_fmt = raw[0,0,16]
# img_fmt += 1
# img_fmt /= 2
# img_fmt *= 255
# print(img_fmt.shape)
# print(img_fmt.max(), img_fmt.min())
# raw_img = Image.fromarray(img_fmt.astype(np.uint8), "L")
# #labels_img = Image.fromarray(labels[0,0,16])
# raw_img.save("raw.png")

# pred_affs1, _ = model1(torch.tensor(raw).to("cuda"))
# pred_affs2, _ = model2(torch.tensor(raw).to("cuda"))
# image_tensors = {"aff": pred_affs1, "raw": raw}
# images = get_2D_snapshot(image_tensors)
# images[0].save("aff_test2.png")
# images[1].save("raw_test2.png")
# 
# zarrs = save_zarr_snapshot(
#     "test.zarr",
#     f"test",
#     image_tensors,
# )

#pred_affs1 = pred_affs1.detach().cpu().numpy()
#pred_affs2 = pred_affs2.detach().cpu().numpy()
#print(pred_affs1.min(), pred_affs1.max())
#print(pred_affs2.min(), pred_affs2.max())
#pred_affs1 = pred_affs1[0,:,2]
#pred_affs2 = pred_affs2[0,:,2]
##print(pred_affs.shape)
#
## pred_affs1 += 1
## pred_affs1 /= 2
#pred_affs1 *= 255
#pred_affs1 = np.transpose(pred_affs1, (1, 2, 0))
#affs_img1 = Image.fromarray(pred_affs1.astype(np.uint8))
#affs_img1.save("affs1.png")
#
#pred_affs2 += 1
#
#pred_affs2 /= 2
#
#pred_affs2 *= 255
#
#pred_affs2 = np.transpose(pred_affs2, (1, 2, 0))
#
#
#affs_img2 = Image.fromarray(pred_affs2.astype(np.uint8))
#affs_img2.save("affs2.png")
