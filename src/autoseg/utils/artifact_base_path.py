import os


def get_artifact_base_path(config=None):
    #path = "./artifacts/"
    path = "/mnt/c/Users/anton/Documents/GitHub/autoseg/src/autoseg/artifacts/"
    if config is not None:
        path += config["model"]["name"] + "/"
    os.makedirs(path, exist_ok=True)
    return path
