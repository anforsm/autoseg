import os


def get_artifact_base_path(config=None):
    path = "./artifacts/"
    if config is not None:
        path = "./artifacts/" + config["model"]["name"] + "/"
    os.makedirs(path, exist_ok=True)
    return path
