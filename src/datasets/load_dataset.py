from pathlib import Path
import os
import fsspec
from .zarr_dataset import ZarrDataset

ROOT_PATH = Path.home() / Path(".cache/autoseg/datasets/")


def get_synapseweb_dataset_names(dataset):
    volume = dataset.split("/")[-1]
    repo_id = "/".join(dataset.split("/")[:-1])
    filename = f"data/{volume}.zarr.zip"
    return repo_id, volume, filename


def download_dataset(dataset="SynapseWeb/kh2015/oblique"):
    if dataset.startswith("SynapseWeb"):
        repo_id, volume, filename = get_synapseweb_dataset_names(dataset)

        from huggingface_hub import hf_hub_download

        os.makedirs(ROOT_PATH.as_posix(), exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=ROOT_PATH / Path(repo_id),
            local_dir_use_symlinks=False,
        )
    else:
        raise ValueError("Only SynapseWeb datasets are supported at the moment")


def dataset_exists(dataset="SynapseWeb/kh2015/oblique"):
    repo_id, _, filename = get_synapseweb_dataset_names(dataset)
    return (ROOT_PATH / Path(repo_id) / Path(filename)).exists()


def load_dataset(dataset="SynapseWeb/kh2015/oblique"):
    repo_id, volume, filename = get_synapseweb_dataset_names(dataset)

    if not dataset_exists(dataset):
        print("Dataset not found on disk, downloading...")
        download_dataset(dataset)

    dataset = ZarrDataset(
        container_path=ROOT_PATH / Path(repo_id) / Path(filename),
        dataset_name="raw/s0",
        num_spatial_dims=3,
        crop_shape=(36, 212, 212),  # (50, 512, 512)
    )
    return dataset


if __name__ == "__main__":
    download_dataset()
