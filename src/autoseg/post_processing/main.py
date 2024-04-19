import sys
import yaml

from autoseg.config import read_config
from autoseg.datasets import get_dataset_path
from autoseg.utils import get_artifact_base_path
from pathlib import Path
import subprocess


# Runs postprocessing for a model
if __name__ == "__main__":
    config = sys.argv[1]
    config = read_config(config)

    with open("post_processing/watershed/config.yaml", "r") as f:
        postproc_yaml_config = yaml.safe_load(f)

    pred_file = (
        Path(
            get_artifact_base_path(config)
            + "UNet_LSD/predictions/step-9900/oblique_prediction.zarr"
        )
        .absolute()
        .as_posix()
    )

    p = postproc_yaml_config
    p["db"]["db_name"] = "anton_UNet_LSD_9900_oblique"

    p["processing"]["extract_fragments"]["affs_file"] = pred_file
    p["processing"]["extract_fragments"]["affs_dataset"] = "preds/affs"
    p["processing"]["extract_fragments"]["fragments_file"] = pred_file
    p["processing"]["extract_fragments"]["fragments_dataset"] = "frags"
    p["processing"]["extract_fragments"]["mask_file"] = None
    p["processing"]["extract_fragments"]["mask_dataset"] = None

    p["processing"]["agglomerate"]["affs_file"] = pred_file
    p["processing"]["agglomerate"]["affs_dataset"] = "preds/affs"
    p["processing"]["agglomerate"]["fragments_file"] = pred_file
    p["processing"]["agglomerate"]["fragments_dataset"] = "frags"

    p["processing"]["find_segments"]["fragments_file"] = pred_file
    p["processing"]["find_segments"]["fragments_dataset"] = "frags"
    p["processing"]["find_segments"]["lut_dir"] = "luts"

    p["processing"]["extract_segmentation"]["fragments_file"] = pred_file
    p["processing"]["extract_segmentation"]["fragments_dataset"] = "frags"
    p["processing"]["extract_segmentation"]["seg_file"] = pred_file
    p["processing"]["extract_segmentation"]["seg_dataset"] = "seg"

    yaml.dump(p, open("post_processing/watershed/temp_config.yaml", "w"))

    subprocess.run(
        [
            "python",
            "post_processing/watershed/02_extract_fragments_blockwise.py",
            "post_processing/watershed/temp_config.yaml",
        ]
    )
    subprocess.run(
        [
            "python",
            "post_processing/watershed/03_agglomerate_blockwise.py",
            "post_processing/watershed/temp_config.yaml",
        ]
    )
    subprocess.run(
        [
            "python",
            "post_processing/watershed/04_find_segments.py",
            "post_processing/watershed/temp_config.yaml",
        ]
    )
    subprocess.run(
        [
            "python",
            "post_processing/watershed/05_extract_segments_from_lut.py",
            "post_processing/watershed/temp_config.yaml",
        ]
    )
