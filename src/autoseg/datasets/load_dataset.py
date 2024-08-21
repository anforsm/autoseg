from pathlib import Path
import os

if not "AUTOSEG_CACHE" in os.environ:
    ROOT_PATH = Path.home() / Path(".cache/autoseg")
else:
    ROOT_PATH = Path(os.environ["AUTOSEG_CACHE"])

DATASET_PATH = ROOT_PATH / Path("datasets/")


def get_synapseweb_dataset_names(dataset):
    volume = dataset.split("/")[-1]
    repo_id = "/".join(dataset.split("/")[:-1])
    if repo_id == "SynapseWeb/team_dentate":
        filename = f"data/{volume}"
        if "." not in volume:
            filename += ".zarr"
    else:
        filename = f"data/{volume}.zarr.zip"
    return repo_id, volume, filename


def download_dataset(dataset="SynapseWeb/kh2015/oblique", force=False):
    if dataset_exists(dataset):
        return
    else:
        print("Dataset not found on disk, downloading...")
    if force:
        print("Downloading dataset")

    if dataset.startswith("SynapseWeb"):
        repo_id, volume, filename = get_synapseweb_dataset_names(dataset)

        from huggingface_hub import hf_hub_download

        os.makedirs(DATASET_PATH.as_posix(), exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=Path(DATASET_PATH / Path(repo_id).as_posix()),
            local_dir_use_symlinks=False,
        )
    else:
        raise ValueError("Only SynapseWeb datasets are supported at the moment")


def get_dataset_path(dataset="SynapseWeb/kh2015/oblique"):
    if not dataset.startswith("SynapseWeb"):
        return Path(dataset)
    repo_id, _, filename = get_synapseweb_dataset_names(dataset)
    return DATASET_PATH / Path(repo_id) / Path(filename)


def dataset_exists(dataset="SynapseWeb/kh2015/oblique"):
    return get_dataset_path(dataset).exists()


if __name__ == "__main__":
    download_dataset()
