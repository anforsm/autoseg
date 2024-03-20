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

from autoseg.models import ExampleModel, ExampleModel2D
from autoseg.config import read_config
from autoseg.datasets.load_dataset import get_dataset_path, download_dataset

# CONFIG_PATH = "examples/kh2015_multisource"
CONFIG_PATH = "examples/2d_multisource"


Z_RES = 50  # nm / px, arbitrary


def copy(block, in_ds, out_ds):
    roi_2d = Roi(block.read_roi.offset[1:], block.read_roi.shape[1:])

    in_array = in_ds[roi_2d]
    in_array_data = in_array.to_ndarray(roi=roi_2d)

    if len(in_array_data.shape) == 3:
        out_array_data = np.expand_dims(in_array_data, axis=1)
    elif len(in_array_data.shape) == 2:
        out_array_data = np.expand_dims(in_array_data, axis=0)

    out_ds[block.write_roi] = out_array_data


def stack_datasets(
    zarr_container: str,
    input_datasets: list[str],
    out_ds_name: str,
    num_workers: int = 100,
):
    in_ds = open_ds(zarr_container, input_datasets[0])

    shape = list(in_ds.shape)
    if len(shape) == 3:
        shape.insert(1, len(input_datasets))
        num_channels = shape[0]
    elif len(shape) == 2:
        shape.insert(0, len(input_datasets))
        num_channels = None
    print(shape)

    voxel_size = list(in_ds.voxel_size)
    voxel_size_3d = Coordinate(
        [
            Z_RES,
        ]
        + voxel_size
    )
    total_3d_roi = Roi((0, 0, 0), Coordinate(shape[-3:]) * voxel_size_3d)

    chunk_shape_2d = list(in_ds.chunk_shape)[-2:]
    block_shape = (
        Coordinate(
            [
                len(input_datasets),
            ]
            + chunk_shape_2d
        )
        * voxel_size_3d
    )

    read_roi = write_roi = Roi((0, 0, 0), block_shape)

    print(chunk_shape_2d, voxel_size_3d, block_shape)

    out_ds = prepare_ds(
        zarr_container,
        out_ds_name,
        total_roi=total_3d_roi,
        voxel_size=voxel_size_3d,
        dtype=np.uint8,
        num_channels=num_channels,
        write_size=write_roi.shape,
        force_exact_write_size=False,
        delete=True,
    )

    for i, in_ds_name in enumerate(input_datasets):
        print(f"Copying {in_ds_name} to {out_ds_name}")

        in_ds = open_ds(zarr_container, in_ds_name)

        total_roi = Roi(
            (Z_RES * i, 0, 0), Coordinate([1, shape[-2], shape[-1]]) * voxel_size_3d
        )

        task = daisy.Task(
            f"StackTask_{i}/{shape[-3]}",
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=lambda b: copy(b, in_ds, out_ds),
            check_function=None,
            num_workers=num_workers,
            read_write_conflict=True,
            fit="shrink",
        )

        done = daisy.run_blockwise([task])

        if not done:
            raise RuntimeError("at least one block failed!")


def predict_zarr(
    input_zarr: zarr.hierarchy.Group, output_zarr, input_image_shape, output_image_shape
):
    input_image_shape = Coordinate(input_image_shape)
    output_image_shape = Coordinate(output_image_shape)

    # voxel_size = Coordinate([50, 2, 2])
    voxel_size = Coordinate([2, 2])

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
        # num_channels=3,
        num_channels=2,
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

        model = ExampleModel2D()
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
        num_workers=50,
        read_write_conflict=True,
        max_retries=5,
        fit="overhang",
    )

    done = daisy.run_blockwise([task])

    if not done:
        raise RuntimeError("Blockwise prediction failed")


config = read_config(CONFIG_PATH)

if __name__ == "__main__":
    num_source_configs = len(config["predict"]["source"])

    for i in range(num_source_configs):
        source_config = config["predict"]["source"][i]
        download_dataset(source_config["path"])
        input_zarr = zarr.open(get_dataset_path(source_config["path"]), mode="r")
        input_zarr = input_zarr[source_config["dataset"]]

        output_config = config["predict"]["output"][0]
        out_ds = output_config["dataset"] + f"/{i}"

        output_zarr = zarr.open(output_config["path"], mode="a")

        output_zarr.create_dataset(
            out_ds,
            shape=input_zarr.shape,
            overwrite=True,
        )
        output_zarr = output_zarr[out_ds]

        predict_zarr(
            input_zarr,
            output_zarr,
            config["training"]["train_dataloader"]["input_image_shape"],
            config["training"]["train_dataloader"]["output_image_shape"],
        )

    # Combine the predictions into one zarr if more than one source
    if num_source_configs > 1:
        stack_datasets(
            output_config["path"],
            [output_config["dataset"] + f"/{i}" for i in range(num_source_configs)],
            output_config["dataset"] + "/combined",
        )
