import torch
from pathlib import Path
from .configurable_unet import ConfigurableUNet
from .configurable_unetr import ConfigurableUNETR
from .unets import UNETR
from transformers import PreTrainedModel
from huggingface_hub import PyTorchModelHubMixin
import json
import os

from autoseg.datasets.load_dataset import ROOT_PATH
from autoseg.utils import get_artifact_base_path

MODELS_PATH = ROOT_PATH / "models"

# Should use https://huggingface.co/docs/transformers/v4.39.0/en/main_classes/model#transformers.PreTrainedModel


class Model(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        model_config = config["model"]
        class_ = model_config["class"]
        super().__init__()

        self.name = model_config["name"]
        self.path = model_config["path"]
        self.hf_path = model_config["hf_path"]
        if "downsample_factors" in model_config["hyperparameters"]:
            model_config["hyperparameters"]["downsample_factors"] = tuple(
                tuple(x) for x in model_config["hyperparameters"]["downsample_factors"]
            )

        if class_ == "UNet":
            self.model = ConfigurableUNet(**model_config["hyperparameters"])
        if class_ == "UNETR":
            self.model = ConfigurableUNETR(
                image_shape=config["model"]["input_image_shape"],
                **model_config["hyperparameters"]
                # input_dim=1,
                # output_dim=12,
                # patch_size=16,
                # embed_dim=32,
                # num_heads=1
            )

        self.config = model_config

    def checkpoint(self, iteration, local_only=True):
        pass

    def save(self, **kwargs):
        self.save_to_local(**kwargs)
        if not self.hf_path is None:
            self.save_to_hf(**kwargs)

    @staticmethod
    def get_subname(**kwargs):
        name = None
        if "step" in kwargs:
            if (
                "overwrite_checkpoints" in kwargs
                and not kwargs["overwrite_checkpoints"]
            ):
                name = f"step-{kwargs['step']}"

        if "save_best" in kwargs and kwargs["save_best"]:
            name = "best"

        return name

    def save_to_hf(self, **kwargs):
        # Saves to huggingface hub, uses branch  that is number of steps
        print("Uploading model to Hugging Face 🤗")
        commit = self.get_subname(**kwargs)
        self.push_to_hub(self.hf_path, commit_message=commit)

    def save_to_local(self, **kwargs):
        subpath_for_checkpoint = self.get_subname(**kwargs)
        path = (
            Path(get_artifact_base_path({"model": {"name": self.name}}))
            / Path(self.path)
            / Path(subpath_for_checkpoint)
        ).as_posix()
        print(path)
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.state_dict(),
            path + "/ckpt.pt",
        )
        # self.save_pretrained(path)

    def get_local_path(self, checkpoint=None, **kwargs):
        if checkpoint is None:
            path = (
                Path(get_artifact_base_path({"model": {"name": self.name}}))
                / Path(self.path)
                / Path("best/ckpt.pt")
            )
        else:
            path = (
                Path(get_artifact_base_path({"model": {"name": self.name}}))
                / Path(self.path)
                / Path(checkpoint)
                / Path("ckpt.pt")
            )
        return path.absolute().as_posix()

    def load(self, **kwargs):
        local_path = self.get_local_path(**kwargs)
        if Path(local_path).exists():
            self.load_from_local(local_path, **kwargs)
        else:
            try:
                self.load_from_hf()
            except:
                pass

    def load_from_hf(self):
        if self.hf_path is not None:
            self.from_pretrained(self.hf_path, cache_dir=MODELS_PATH)

    def load_from_local(self, local_path, checkpoint=None, **kwargs):
        print(local_path)
        weights = torch.load(local_path)
        self.load_state_dict(weights)

    def forward(self, input):
        return self.model(input)
