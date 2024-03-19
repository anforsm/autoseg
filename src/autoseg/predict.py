import multiprocessing

multiprocessing.set_start_method("fork")

import torch

from funlib.geometry import Roi, Coordinate
from funlib.persistence import prepare_ds
from pathlib import Path
import daisy
import json
import logging
import os
import subprocess
import time
import gunpowder as gp
import zarr

from autoseg.models import ExampleModel
from autoseg.config import read_config
from autoseg.datasets.load_dataset import get_dataset_path, download_dataset

CONFIG_PATH = "examples/kh2015_multisource"


def predict_zarr(
    input_zarr: zarr.hierarchy.Group, output_zarr, input_image_shape, output_image_shape
):
    input_image_shape = Coordinate(input_image_shape)
    output_image_shape = Coordinate(output_image_shape)

    voxel_size = Coordinate([50, 2, 2])

    input_image_size = input_image_shape * voxel_size
    output_image_size = output_image_shape * voxel_size
    context = (input_image_size - output_image_size) // 2

    raw = gp.ArrayKey("RAW")
    affs = gp.ArrayKey("AFFS")

    path = input_zarr.store.path
    ds = input_zarr.path

    source_node = gp.ZarrSource(
        path, {raw: ds}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source_node):
        input_roi = source_node.spec[raw].roi
        output_roi = source_node.spec[raw].roi.grow(-context, -context)

    ndims = len(input_roi.shape)
    block_read_roi = Roi((0,) * ndims, input_image_size) - context
    block_write_roi = Roi((0,) * ndims, output_image_size)

    o_path = output_zarr.store.path
    o_ds = output_zarr.path

    prepare_ds(
        filename=o_path,
        ds_name=o_ds,
        total_roi=output_roi,
        voxel_size=voxel_size,
        write_size=block_write_roi.shape,
        delete=True,
        num_channels=3,
        dtype="uint8",
    )

    logging.info("Starting workers...")

    def predict_gunpowder():
        def get_device_id():
            try:
                device_id = (
                    int(daisy.Context.from_env()["worker_id"])
                    % torch.cuda.device_count()
                )
            except Exception as e:
                logging.warning(f"Could not get device id from environment: {e}")
                device_id = 0
            return f"cuda:{device_id}"

        model = ExampleModel()
        model.eval()
        model.load_state_dict(
            torch.load("out/latest_model3.pt", map_location=get_device_id())
        )

        chunk_request = gp.BatchRequest()
        chunk_request.add(raw, input_image_size)
        chunk_request.add(affs, output_image_size)

        pipeline = gp.ZarrSource(
            path, {raw: ds}, {raw: gp.ArraySpec(interpolatable=True)}
        )
        pipeline += gp.Normalize(raw)
        pipeline += gp.IntensityScaleShift(raw, 2, -1)
        pipeline += gp.Pad(raw, None)  # Not sure if needed
        pipeline += gp.Unsqueeze([raw])  # Add 1d channel dim
        pipeline += gp.Unsqueeze([raw])  # Add 1d batch dim
        pipeline += gp.torch.Predict(
            model,
            inputs={"input": raw},
            outputs={0: affs},
            device=get_device_id(),
        )
        pipeline += gp.Squeeze([affs])  # Remove 1d batch dim
        pipeline += gp.IntensityScaleShift(affs, 255, 0)
        pipeline += gp.ZarrWrite(
            output_dir="/".join(o_path.split("/")[:-1]),
            output_filename=o_path.split("/")[-1],
            dataset_names={
                affs: o_ds,
            },
        )
        pipeline += gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: "read_roi",
                affs: "write_roi",
            },
            num_workers=1,
        )

        with gp.build(pipeline):
            pipeline.request_batch(gp.BatchRequest())

    task = daisy.Task(
        "PredictBlockwiseTask",
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=predict_gunpowder,
        check_function=None,
        num_workers=5,
        read_write_conflict=True,
        max_retries=5,
        fit="overhang",
    )

    done = daisy.run_blockwise([task])

    if not done:
        raise RuntimeError("Blockwise prediction failed")


config = read_config(CONFIG_PATH)

if __name__ == "__main__":
    source_config = config["predict"]["source"][0]
    download_dataset(source_config["path"])
    input_zarr = zarr.open(get_dataset_path(source_config["path"]), mode="r")
    input_zarr = input_zarr[source_config["dataset"]]

    output_config = config["predict"]["output"][0]
    output_zarr = zarr.open(output_config["path"])
    output_zarr = output_zarr[output_config["dataset"]]

    predict_zarr(
        input_zarr,
        output_zarr,
        config["training"]["train_dataloader"]["input_image_shape"],
        config["training"]["train_dataloader"]["output_image_shape"],
    )
