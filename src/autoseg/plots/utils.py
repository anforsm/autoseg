import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import wandb
import pandas as pd
import seaborn as sns
from collections import defaultdict
from autoseg.utils import get_artifact_base_path
from pathlib import Path


class DefaultDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory(key)


c = DefaultDict(lambda x: x)

c["skel_mods"] = "Total Edits"
c["checkpoints"] = "Update Steps"
c["voi_sum"] = "VOI"
c["nvi_sum"] = "NVI Sum"
c["nvi_split"] = "NVI Split"
c["nvi_merge"] = "NVI Merge"
c["mods_per_length"] = "Edits per Path Length (nm)"
c["mods_per_obj"] = "Edits per Object"
c["skel_split_per_seg"] = "Split Edits per Segment"
c["skel_merge_per_seg"] = "Merge Edits per Segment"


def get_vals_for_metric(metric, files):
    xs = []
    vals = []
    for f in sorted(files, key=lambda x: int(x.split("/")[-2].split("-")[1])):
        xs.append(int(f.split("/")[-2].split("-")[1]))
        results = json.load(open(f))
        vals.append(results["best_edits"][metric])
    return xs, vals


def get_metrics_for_model(model_name):
    if model_name.endswith(" old"):
        model_name = model_name.replace(" old", "")
        path = f"/home/anton/github/autoseg/src/autoseg/artifacts/{model_name.replace(' ', '_')}/results_old"
    else:
        path = f"/home/anton/github/autoseg/src/autoseg/artifacts/{model_name.replace(' ', '_')}/results"
    # if not glob.glob(path):
    #  path = f"/home/anton/github/autoseg/src/autoseg/artifacts/{model_name.replace(' ', '_')}/results_old"

    files = glob.glob(path + "/step-*/result.json")
    for f in files:
        if "step-0" in f:
            files.remove(f)
    print(files)

    xs, merges = get_vals_for_metric("total_merges_needed_to_fix_splits", files)
    _, splits = get_vals_for_metric("total_splits_needed_to_fix_merges", files)
    skel_mods = np.array(merges) + np.array(splits)
    xs = np.array(xs)

    voi_sum = np.array(get_vals_for_metric("voi_sum", files)[1])
    nvi_sum = np.array(get_vals_for_metric("nvi_sum", files)[1])
    path_length = get_vals_for_metric("total path length", files)[1]
    return {
        "checkpoints": xs,
        "merges": merges,
        "splits": splits,
        "skel_mods": skel_mods,
        "skel_splits": get_vals_for_metric("total_splits_needed_to_fix_merges", files)[
            1
        ],
        "skel_merges": get_vals_for_metric("total_merges_needed_to_fix_splits", files)[
            1
        ],
        "mods_per_length": skel_mods / path_length,
        "mods_per_obj": skel_mods
        / get_vals_for_metric("number_of_components", files)[1],
        "skel_split_per_seg": get_vals_for_metric(
            "average_splits_needed_to_fix_merges", files
        )[1],
        "skel_merge_per_seg": get_vals_for_metric(
            "average_merges_needed_to_fix_splits", files
        )[1],
        "voi_sum": voi_sum,
        "voi_merge": get_vals_for_metric("voi_merge", files)[1],
        "voi_split": get_vals_for_metric("voi_split", files)[1],
        "nvi_sum": nvi_sum,
        "nvi_merge": get_vals_for_metric("nvi_merge", files)[1],
        "nvi_split": get_vals_for_metric("nvi_split", files)[1],
        "total_path_length": path_length,
        "threshold": get_vals_for_metric("threshold", files)[1],
    }


def create_dataframe(models):
    metrics_example = get_metrics_for_model(models[list(models.keys())[0]][0])
    df = pd.DataFrame(
        columns=["Model Type", "Run", "Run Name"] + list(metrics_example.keys())
    )
    for group_name, group in models.items():
        for model_i, model in enumerate(group):
            metrics = get_metrics_for_model(model)
            for i, checkpoint in enumerate(metrics["checkpoints"]):
                df.loc[len(df)] = [group_name, model_i, model] + [
                    metrics[key][i] for key in metrics.keys()
                ]
    return df


import gspread
import gdown
import zipfile
import os
import re
from google.oauth2.service_account import Credentials


def get_file_id(url):
    # Handle different types of Google Drive links
    file_id = None
    patterns = [
        r"https://drive\.google\.com/file/d/([\w-]+)",
        r"https://drive\.google\.com/open\?id=([\w-]+)",
        r"https://drive\.google\.com/uc\?id=([\w-]+)",
        r"https://docs\.google\.com/uc\?id=([\w-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            break
    return file_id


def download_results(config):
    sheet_url = "https://docs.google.com/spreadsheets/d/1KDPX08F_CjH9GCaCom_WpnSfqSdDkYehZQ8OVxHx9Ow/edit?gid=0#gid=0"

    # Connect to the Google Sheet
    client = gspread.service_account(
        filename="/home/anton/github/autoseg/src/autoseg/service_account.json"
    )
    sheet = client.open_by_url(sheet_url).worksheet("Sheet1")
    run_name = config["model"]["name"]

    # Find the row for the given run_name
    all_values = sheet.get_all_values()
    row = next((row for row in all_values if row[0] == run_name), None)

    if not row:
        print(f"No results found for run: {run_name}")
        return

    # Extract the results link
    results_link = row[9]

    if not results_link:
        print(f"No results link found for run: {run_name}")
        return

    # Get file ID from the link
    file_id = get_file_id(results_link)
    if not file_id:
        print(f"Invalid Google Drive link: {results_link}")
        return

    # Download the file using gdown
    base_path = (
        (
            Path("/home/anton/github/autoseg/src/autoseg")
            / Path(get_artifact_base_path(config))
        )
        .absolute()
        .as_posix()
    )
    os.makedirs(base_path, exist_ok=True)
    zip_path = f"{base_path}/results.zip"
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Attempting to download from: {url} to: {zip_path}")
    try:
        output = gdown.download(url, zip_path, quiet=False)
        if output is None:
            print(f"Failed to download results for run: {run_name}")
            return
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return

    if not os.path.exists(zip_path):
        print(f"Failed to download results for run: {run_name}")
        return

    # Extract the zip file
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            path = f"{base_path}/{config['evaluation']['results_dir']}"
            zip_ref.extractall(path)
        print(f"Results extracted to {path}")
    except zipfile.BadZipFile:
        print(
            f"The downloaded file is not a valid zip file. It may be an HTML error page."
        )
        with open(zip_path, "r") as f:
            print(f"File contents: {f.read()[:1000]}...")  # Print first 1000 characters
        return
    except Exception as e:
        print(f"Error extracting zip file: {str(e)}")
        return

    # Remove the zip file
    os.remove(zip_path)

    print(f"Results downloaded and extracted to {path}")


# Example usage
if __name__ == "__main__":
    download_results("v2_UNet_run_1")
