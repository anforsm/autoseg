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
import glob
from pathlib import Path

from typing import List
from typing import TypedDict

# import logging
# logging.basicConfig(level=logging.INFO)
from autoseg.utils import get_artifact_base_path

from predict_blockwise import predict_blockwise


class ZarrConfig(TypedDict):
    path: str
    dataset: str


from autoseg.models import ExampleModel, ExampleModel2D, Model
from autoseg.config import read_config
from autoseg.datasets.load_dataset import get_dataset_path, download_dataset
from autoseg.datasets.utils import get_voxel_size, get_shape

# CONFIG_PATH = "defaults"
CONFIG_PATH = "autoseg/examples/lsd"


Z_RES = 50  # nm / px, arbitrary


def start_worker():
    worker_id = daisy.Context.from_env()["worker_id"]
    task_id = daisy.Context.from_env()["task_id"]

    print(f"worker {worker_id} started for task {task_id}...")
    logging.log(logging.WARNING, f"worker {worker_id} started for task {task_id}...")

    subprocess.run(["python", "predict_blockwise.py"])


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


def list_to_array_keys(list_):
    return [gp.ArrayKey(f.upper()) for f in list_]


def predict_zarr(
    input_zarr: ZarrConfig,
    output_zarrs: List[ZarrConfig],
    input_image_shape,
    output_image_shape,
    model_outputs,
    num_workers,
    model_checkpoint_path,
    shape_increase,
    config,
):
    shape_increase = Coordinate(shape_increase)
    input_image_shape = Coordinate(input_image_shape) + shape_increase
    output_image_shape = Coordinate(output_image_shape) + shape_increase

    # voxel_size = Coordinate([50, 2, 2])
    voxel_size = Coordinate(get_voxel_size(input_zarr["path"], input_zarr["dataset"]))

    input_image_size = input_image_shape * voxel_size
    output_image_size = output_image_shape * voxel_size
    context = (input_image_size - output_image_size) // 2

    raw = gp.ArrayKey("RAW")
    # array_keys = list_to_array_keys(model_outputs)

    path = input_zarr["path"]
    ds = input_zarr["dataset"]

    source_node = gp.ZarrSource(
        path, {raw: ds}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source_node):
        input_roi = source_node.spec[raw].roi.grow(context, context)
        output_roi = source_node.spec[raw].roi

    ndims = len(input_roi.shape)
    block_read_roi = Roi((0,) * ndims, input_image_size) - context
    block_write_roi = Roi((0,) * ndims, output_image_size)

    o_path = output_zarrs[0]["path"]
    o_datasets = [output_zarr["dataset"] for output_zarr in output_zarrs]

    conf = {
        "config": config,
        "array_keys": model_outputs,
        "input_path": path.as_posix(),
        "input_dataset": ds,
        "output_path": o_path,
        "output_datasets": o_datasets,
        "input_image_size": input_image_size,
        "output_image_size": output_image_size,
        "output_roi": (output_roi.offset, output_roi.shape),
        "multi_gpu": True,
        "num_workers": config["predict"]["num_workers"],
        "model_checkpoint_path": model_checkpoint_path,
    }

    if not config["predict"]["multi_gpu"]:
        conf["multi_gpu"] = False
        predict_blockwise(**conf)

    if config["predict"]["multi_gpu"]:
        # def f():
        #    predict_gunpowder(
        #        multi_gpu=True, num_workers=config["predict"]["num_workers"]
        #    )

        json.dump(conf, open("conf.json", "w"))

        task = daisy.Task(
            "PredictBlockwiseTask",
            input_roi,
            block_read_roi,
            block_write_roi,
            process_function=start_worker,
            check_function=None,
            num_workers=num_workers,
            read_write_conflict=False,
            max_retries=5,
            fit="overhang",
        )

        done = daisy.run_blockwise([task])

        if not done:
            raise RuntimeError("Blockwise prediction failed")


def get_checkpoint_paths(config):
    paths = []
    base_path = get_artifact_base_path(config) + config["model"]["path"]

    checkpoint_steps = glob.glob(base_path + "/step-*")
    checkpoint_steps = [int(step.split("-")[-1]) for step in checkpoint_steps]

    if (
        "predict_with_best_checkpoint" in config["predict"]
        and config["predict"]["predict_with_best_checkpoint"]
    ):
        paths.append("best")
    if (
        "predict_with_last_checkpoint" in config["predict"]
        and config["predict"]["predict_with_last_checkpoint"]
    ):
        latest = max(checkpoint_steps)
        paths.append(f"step-{latest}")

    if (
        "predict_with_every_n_checkpoint" in config["predict"]
        and config["predict"]["predict_with_every_n_checkpoint"] == 0
    ):
        return paths

    for i in range(
        0, len(checkpoint_steps), config["predict"]["predict_with_every_n_checkpoint"]
    ):
        path = f"step-{checkpoint_steps[i]}"
        if not path in paths:
            paths.append(path)
    return paths


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = CONFIG_PATH

    config = read_config(config_path)

    num_source_configs = len(config["predict"]["datasets"])
    model_outputs = config["training"]["model_outputs"]

    paths = get_checkpoint_paths(config)

    for model_checkpoint_path in paths:
        model = Model(config)
        model.load(checkpoint=model_checkpoint_path)

        for i in range(num_source_configs):
            shape_increase = Coordinate(
                config["predict"]["datasets"][i]["shape_increase"]
            )
            source_config = config["predict"]["datasets"][i]["source"]
            download_dataset(source_config["path"])
            # input_zarr = zarr.open(get_dataset_path(source_config["path"]), mode="r")
            # input_zarr = input_zarr[source_config["dataset"]]
            input_zarr: ZarrConfig = {
                "path": get_dataset_path(source_config["path"]),
                "dataset": source_config["dataset"],
            }

            resolution = get_voxel_size(input_zarr["path"], input_zarr["dataset"])

            output_zarrs = []
            # If model has multiple outputs, e.g. affs and lsd
            for j, output_config in enumerate(
                config["predict"]["datasets"][i]["output"]
            ):
                # Only if stacking inputs
                # out_ds = output_config["dataset"] + (
                #    f"/{j}" if num_source_configs > 1 else ""
                # )
                out_ds = output_config["dataset"]

                output_size = (
                    Coordinate(config["model"]["output_image_shape"]) + shape_increase
                ) * Coordinate(resolution)
                shape = get_shape(input_zarr["path"], input_zarr["dataset"])
                print(output_size, shape, resolution)

                out_path = (
                    Path(get_artifact_base_path(config))
                    / Path("predictions")
                    / Path(model_checkpoint_path)
                    / Path(output_config["path"])
                )
                if "mask" in config["predict"]["datasets"][i]:
                    labels_mask = zarr.open(
                        get_dataset_path(
                            config["predict"]["datasets"][i]["mask"]["path"]
                        ),
                        mode="r",
                    )
                    labels_mask = labels_mask[
                        config["predict"]["datasets"][i]["mask"]["dataset"]
                    ]
                    roi = gp.Roi(
                        labels_mask.attrs["offset"],
                        Coordinate(labels_mask.shape) * Coordinate(resolution),
                    )
                else:
                    raw = zarr.open(input_zarr["path"], mode="r")
                    raw = raw[input_zarr["dataset"]]
                    roi = gp.Roi(
                        raw.attrs["offset"],
                        Coordinate(shape) * Coordinate(resolution),
                    )
                out_path = out_path.resolve().as_posix()
                prepare_ds(
                    filename=out_path,
                    ds_name=out_ds,
                    total_roi=roi,
                    write_size=output_size,
                    delete=True,
                    voxel_size=Coordinate(resolution),
                    num_channels=output_config["num_channels"],
                    compressor={"id": "blosc"},
                    # force_exact_write_size=True,
                    dtype="uint8",
                )
                print(out_path, out_ds)
                output_zarrs.append(
                    {
                        "path": out_path,
                        "dataset": out_ds,
                    }
                )

                # output_zarr = zarr.open(output_config["path"], mode="a")
                # print(list(output_zarr))
                # print(list(output_zarr["preds"]))
                # output_zarr = output_zarr[out_ds]
                # output_zarrs.append(output_zarr)

                # output_zarr.create_group(
                #    out_ds,
                #    #shape=input_zarr.shape,
                #    overwrite=True,
                # )
                # output_zarr = output_zarr[out_ds]

            predict_zarr(
                input_zarr,
                output_zarrs,
                config["model"]["input_image_shape"],
                config["model"]["output_image_shape"],
                model_outputs,
                config["predict"]["num_workers"],
                model_checkpoint_path,
                shape_increase,
                config,
            )

        # Combine the predictions into one zarr if more than one source
        # if num_source_configs > 1:
        #     stack_datasets(
        #         output_config["path"],
        #         [output_config["dataset"] + f"/{i}" for i in range(num_source_configs)],
        #         output_config["dataset"] + "/combined",
        #     )
