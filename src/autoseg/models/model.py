import torch
from pathlib import Path
from .configurable_unet import ConfigurableUNet
from transformers import PreTrainedModel
from huggingface_hub import PyTorchModelHubMixin

from autoseg.datasets.load_dataset import ROOT_PATH

MODELS_PATH = ROOT_PATH / "models"

# Should use https://huggingface.co/docs/transformers/v4.39.0/en/main_classes/model#transformers.PreTrainedModel


class Model(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        model_config = config["model"]
        self.name = model_config["name"]
        self.path = model_config["path"]
        self.hf_path = model_config["hf_path"]
        model_config["hyperparameters"]["downsample_factors"] = tuple(
            tuple(x) for x in model_config["hyperparameters"]["downsample_factors"]
        )

        self.model = ConfigurableUNet(**model_config["hyperparameters"])

        self.config = model_config

    def checkpoint(self, iteration, local_only=True):
        pass

    def save(self):
        self.save_to_local()
        if not self.hf_path is None:
            self.save_to_hf()

    def save_to_hf(self):
        self.push_to_hub(self.hf_path)

    def save_to_local(self):
        self.save_pretrained(self.path)
        pass

    def load(self):
        if Path(self.path).exists():
            print("Loading from local")
            self.load_from_local()
        else:
            try:
                self.load_from_hf()
            except:
                pass

    def load_from_hf(self):
        self.from_pretrained(self.hf_path, cache_dir=MODELS_PATH)

    def load_from_local(self):
        self.from_pretrained(Path(self.hf_path), cache_dir=MODELS_PATH)

    def forward(self, input):
        return self.model(input)
