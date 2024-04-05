from PIL.Image import Image
from collections import Iterable


class Logger:
    def __init__(self, name=None, provider="wandb", config=None):
        if not isinstance(provider, list):
            provider = [provider]

        self.providers = provider
        self.current_data = {}
        if self.name is None:
            try:
                self.name = config["model"]["name"]
            except KeyError:
                pass
        self.name = name

        self.config = config

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

        wandb.init(
            name=self.name + " " + wandb.util.generate_id(),
            project="autoseg",
            config=self.config,
        )

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
