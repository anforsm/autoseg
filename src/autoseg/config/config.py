import os
import json
import _jsonnet


def get_curr_dir():
    return os.path.dirname(os.path.realpath(__file__))


def read_config(path=None):
    if path is None or path == "defaults":
        path = get_curr_dir() + "configs/defaults.jsonnet"
    if path.startswith("examples/"):
        path = get_curr_dir() + "/configs/" + path

    if path.endswith(".json"):
        path = path.replace(".json", ".jsonnet")

    if not path.endswith(".jsonnet"):
        path = path + ".jsonnet"

    def cb(dir_, rel):
        if not rel.endswith(".jsonnet"):
            rel = rel + ".jsonnet"

        if rel.startswith("autoseg"):
            new_dir = os.path.join(get_curr_dir(), rel.replace("autoseg/", "configs/"))
            with open(new_dir, "r") as f:
                content = f.read().encode()
            return new_dir, content

    return json.loads(_jsonnet.evaluate_file(path, import_callback=cb))
