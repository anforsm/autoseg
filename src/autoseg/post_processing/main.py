import sys
import yaml
import glob
from tqdm import tqdm

from autoseg.config import read_config
from autoseg.datasets import get_dataset_path
from autoseg.utils import get_artifact_base_path
from autoseg.datasets.utils import get_shape, get_voxel_size
from pathlib import Path
from funlib.geometry import Coordinate
import subprocess


# Runs postprocessing for a model
if __name__ == "__main__":
    config = sys.argv[1]
    config = read_config(config)

    for i in range(len(config["predict"]["datasets"])):
        with open("post_processing/watershed/config.yaml", "r") as f:
            postproc_yaml_config = yaml.safe_load(f)

        # ASSUMES AFFS IS 0:th OUTPUT
        aff_zarrs = glob.glob(
            f"{Path(get_artifact_base_path(config)).absolute()}/predictions/step-*/{config['predict']['datasets'][i]['output'][0]['path']}"
        )
        # print(f"{Path(get_artifact_base_path(config)).absolute()}/predictions/step-*/oblique_prediction.zarr")

        for pred_file in tqdm(aff_zarrs):
            shape = list(get_shape(pred_file, "preds/affs"))
            if len(shape) == 4:
                shape = shape[1:]
            size = list(
                Coordinate(shape) * Coordinate(get_voxel_size(pred_file, "preds/affs"))
            )
            print(size)

            p = postproc_yaml_config
            # p["db"][
            #    "db_name"
            # ] = f"anton_{config['model']['name']}_{pred_file.split('step-')[1].split('/')[0]}".lower()

            p["db"][
                "db_name"
            ] = f"anton_{config['model']['name']}_{pred_file.split('step-')[1].split('/')[0]}_{config['predict']['datasets'][i]['output'][0]['path'].split('.zarr')[0]}".lower()

            print(p["db"]["db_name"])
            # print(p["processing"]["extract_fra"])

            p["processing"]["extract_fragments"]["affs_file"] = pred_file
            p["processing"]["extract_fragments"]["affs_dataset"] = "preds/affs"
            p["processing"]["extract_fragments"]["fragments_file"] = pred_file
            p["processing"]["extract_fragments"]["fragments_dataset"] = "frags"
            p["processing"]["extract_fragments"]["roi_shape"] = None
            p["processing"]["extract_fragments"]["roi_offset"] = None
            p["processing"]["extract_fragments"]["mask_file"] = None
            p["processing"]["extract_fragments"]["mask_dataset"] = None

            if (
                "mask" in config["predict"]["datasets"][i]
                and config["predict"]["datasets"][i]["mask"] is not None
                and not (
                    "use_in_postproc" in config["predict"]["datasets"][i]["mask"]
                    and not config["predict"]["datasets"][i]["mask"]["use_in_postproc"]
                )
            ):
                p["processing"]["extract_fragments"]["mask_file"] = (
                    Path(
                        get_dataset_path(
                            config["predict"]["datasets"][i]["mask"]["path"]
                        )
                    )
                    .absolute()
                    .as_posix()
                )
                p["processing"]["extract_fragments"]["mask_dataset"] = config[
                    "predict"
                ]["datasets"][i]["mask"]["dataset"]

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

            config_path = (
                f"post_processing/watershed/temp_configs/{config['model']['name']}.yaml"
            )

            yaml.dump(p, open(config_path, "w"))

            subprocess.run(
                [
                    "python",
                    "post_processing/watershed/02_extract_fragments_blockwise.py",
                    config_path,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "python",
                    "post_processing/watershed/03_agglomerate_blockwise.py",
                    config_path,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "python",
                    "post_processing/watershed/04_find_segments.py",
                    config_path,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "python",
                    "post_processing/watershed/05_extract_segments_from_lut.py",
                    config_path,
                ],
                check=True,
            )
