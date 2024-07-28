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

from typing import List
from typing import TypedDict


class ZarrConfig(TypedDict):
    path: str
    dataset: str


from autoseg.models import ExampleModel, ExampleModel2D, Model
from autoseg.config import read_config
from autoseg.datasets.load_dataset import get_dataset_path, download_dataset
from autoseg.datasets.utils import get_voxel_size, get_shape


def list_to_array_keys(list_):
    return [gp.ArrayKey(f.upper()) for f in list_]


def predict_blockwise(
    config,
    array_keys,
    input_path,
    input_dataset,
    output_path,
    output_datasets,
    input_image_size,
    output_image_size,
    output_roi,
    multi_gpu=False,
    model_checkpoint_path=None,
    num_workers=10,
):
    output_roi = Roi(output_roi[0], output_roi[1])

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    def get_device_id():
        # return "cuda"
        if not multi_gpu:
            return "cuda:0"
        try:
            device_id = (
                int(daisy.Context.from_env()["worker_id"]) % torch.cuda.device_count()
            )
        except Exception as e:
            logging.warning(f"Could not get device id from environment: {e}")
            device_id = 0
        return f"cuda:{device_id}"

    array_keys = list_to_array_keys(array_keys)

    model = Model(config)
    model.load(checkpoint=model_checkpoint_path)
    model = model.model
    model = model.to(get_device_id())
    model.eval()

    chunk_request = gp.BatchRequest()
    raw = gp.ArrayKey("RAW")
    chunk_request.add(raw, input_image_size)
    print("sizes")
    print(input_image_size)
    print(output_image_size)
    for ak in array_keys:
        chunk_request.add(ak, output_image_size)

    pipeline = gp.ZarrSource(
        input_path, {raw: input_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )
    pipeline += gp.Normalize(raw)
    pipeline += gp.IntensityScaleShift(raw, 2, -1)
    pipeline += gp.Pad(raw, None, mode="reflect")
    pipeline += gp.Unsqueeze([raw])  # Add 1d channel dim
    pipeline += gp.Unsqueeze([raw])  # Add 1d batch dim
    # pipeline += gp.PreCache(
    #    cache_size=10,
    #    num_workers=10
    # )
    pipeline += gp.torch.Predict(
        model,
        inputs={"input": raw},
        outputs={i: ak for i, ak in enumerate(array_keys)},
        device=get_device_id(),
        array_specs={ak: gp.ArraySpec(roi=output_roi) for ak in array_keys}
        if not multi_gpu
        else None,
    )
    pipeline += gp.Squeeze(array_keys)  # Remove 1d batch dim

    for ak in array_keys:
        pipeline += gp.IntensityScaleShift(ak, 255, 0)

    output_dir = Path(output_path).parent.as_posix()
    output_filename = output_path
    pipeline += gp.ZarrWrite(
        store=output_path,
        # output_dir=output_dir,
        # output_filename=output_filename,
        dataset_names={ak: o_ds for ak, o_ds in zip(array_keys, output_datasets)},
    )
    if multi_gpu:
        pipeline += gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                raw: "read_roi",
                **{ak: "write_roi" for ak in array_keys},
            },
            num_workers=1,
        )
    else:
        pass
        # pipeline += gp.PrintProfilingStats()
        pipeline += gp.Scan(chunk_request)

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())


if __name__ == "__main__":
    with open("conf.json", "r") as f:
        config = json.load(f)

    predict_blockwise(**config)
