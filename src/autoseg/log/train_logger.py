from PIL.Image import Image
from collections import Iterable


class Logger:
    def __init__(self, provider="wandb"):
        if not isinstance(provider, list):
            provider = [provider]

        self.providers = provider
        self.current_data = {}

        for provider in self.providers:
            if provider == "wandb":
                self.init_wandb()
            elif provider == "tensorboard":
                self.init_tensorboard()
            else:
                raise ValueError(f"Unknown provider {provider}")

    def init_wandb(self):
        global wandb
        import wandb

        wandb.init(project="autoseg")

    def push(self, data):
        self.current_data.update(data)

    def log(self):
        for provider in self.providers:
            if provider == "wandb":
                self.log_to_wandb()
            elif provider == "tensorboard":
                self.log_to_tensorboard()
            else:
                raise ValueError(f"Unknown provider {provider}")
        self.current_data = {}

    def log_to_wandb(self):
        def wandb_formatter(val):
            if isinstance(val, Image):
                if val.mode == "F":
                    val = val.convert("L")
                return wandb.Image(val)
            return val

        recurse_json(self.current_data, wandb_formatter)

        wandb.log(self.current_data)

    def log_to_tensorboard(self):
        pass


def recurse_json(json, func):
    for key, value in json.items():
        if isinstance(value, dict):
            recurse_json(value, func)
        elif isinstance(value, Iterable) and not isinstance(value, str):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    recurse_json(item, func)
                else:
                    json[key][i] = func(item)
        else:
            json[key] = func(value)
