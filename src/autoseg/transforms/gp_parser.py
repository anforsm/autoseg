import gunpowder as gp
from typing import Dict, List, Union
import json
from functools import cmp_to_key

from autoseg.datasets.load_dataset import get_dataset_path, download_dataset
from autoseg.datasets.utils.zarr_utils import get_voxel_size
import autoseg.transforms.gp_nodes as cgp


def snake_case_to_camel_case(snake_str):
    components = snake_str.split("_")
    return components[0].title() + "".join(x.title() for x in components[1:])


class GunpowderParser:
    def __init__(self, config):
        """Parse a Gunpowder config file into Gunpowder nodes.

        Config file definition

        source_name: str
        config: Dict[source_name, List[GPCollection]]
        GPCollection: List[GPCollection]
        GPCollection: Node
        Node: Dict[NodeName, NodeArgs]
        Node: str (in caps, parsed as an ArrayKey)
        NodeName: str
        NodeArgs: Dict[str, NodeArgValue]
        NodeArgValue: Union[
          str,
          int,
          float,
          bool,
          list,
          Dict,
          Node,
        ]
        """
        self.config = config
        self._array_keys = set()
        self.output_array_keys = []
        self.voxel_sizes = []

    @property
    def voxel_size(self):
        return self.voxel_sizes[0]

    @property
    def array_keys(self):
        return list(sorted(list(self._array_keys), key=lambda ak: ak.identifier))

    def flatten_nodes(self, nodes):  # -> Union[gp.Pipeline, Tuple[gp.Pipeline]]:
        """Flatten a list of nodes.

        If it is a list of nodes or tuples, combine them into a pipeline
        If it is a list of pipelines, convert them into a tuple of pipelines
        """
        # Case of a single node
        if isinstance(nodes, gp.BatchProvider):
            return nodes

        # Base case, start adding nodes to a pipeline
        if isinstance(nodes[0], gp.BatchProvider) or isinstance(nodes[0], tuple):
            pipeline = None
            for node in nodes:
                if pipeline is None:
                    pipeline = node
                else:
                    pipeline += node
            return pipeline

        if isinstance(nodes[0], list):
            return tuple([self.flatten_nodes(node) for node in nodes])

    def parse_config(self):
        config = self.config
        self.nodes = []
        source_names = config["_order"]
        self.output_array_keys = [self.parse_node(ak) for ak in config["_outputs"]]

        for source_name in source_names:
            if not source_name in config:
                print(f"WARN: {source_name} not in config")
                continue
            for gp_collection in config[source_name]:
                collection = self.parse_gp_collection(gp_collection)
                self.nodes.append(collection)

        flattened_nodes = self.flatten_nodes(self.nodes)

        if isinstance(flattened_nodes, gp.Pipeline):
            return flattened_nodes

        pipeline = None
        for node in flattened_nodes:
            if pipeline is None:
                pipeline = node
            else:
                pipeline += node
        return pipeline

    def parse_gp_collection(self, gp_collection):
        if isinstance(gp_collection, list):
            nodes = []
            for gp_collection2 in gp_collection:
                nodes.append(self.parse_gp_collection(gp_collection2))
            return nodes
        else:
            return self.parse_node(gp_collection)

    def parse_node(self, node):
        if isinstance(node, str) and node.isupper():
            return self.parse_node({"array_key": {"identifier": node}})
        elif isinstance(node, str) and node[0] == "_":
            return self.parse_node({"array_key": {"identifier": node[1:].upper()}})
        elif isinstance(node, dict):
            node_name, node_args = list(node.items())[0]
            node_name = snake_case_to_camel_case(node_name)
            if node_name == "Coordinate":
                return gp.Coordinate(node_args["_positional"])

            if node_name == "ArrayKey":
                # otherwise we get infinite recursion
                # ArrayKey only takes one arg which is the identifier (a string)
                # thus we know we dont have to further parse the
                # node args
                kwargs = node_args
            else:
                kwargs = self.parse_node_args(node_args)

            if node_name == "ZarrSource":
                download_dataset(kwargs["store"])
                kwargs["store"] = get_dataset_path(kwargs["store"])

                for dataset in kwargs["datasets"].values():
                    self.voxel_sizes.append(
                        get_voxel_size(path=kwargs["store"].as_posix(), ds=dataset)
                    )
                if not all(self.voxel_sizes[0] == i for i in self.voxel_sizes):
                    print(
                        "WARNING: Voxel sizes are not the same for all datasets. Using",
                        self.voxel_sizes[0],
                    )

                # self.sources.add()

            if node_name == "Pad":
                if "pad" in kwargs and kwargs["pad"] is not None:
                    kwargs["pad"] = gp.Coordinate(kwargs["pad"])

            if node_name == "DownSample" or node_name == "UpSample":
                if "factor" in kwargs:
                    kwargs["factor"] = tuple(kwargs["factor"])

            node = self.get_node(node_name)(**kwargs)

            if node_name == "ArrayKey":
                self._array_keys.add(node)

            return node

    def parse_node_args(self, node_args):
        kwargs = {}
        for key, value in node_args.items():
            kwargs[key] = self.parse_node_arg_value(value)
        return kwargs

    def parse_node_arg_value(self, value):
        if isinstance(value, list):
            return [self.parse_node_arg_value(v) for v in value]
        elif self.is_gunpowder_node(value):
            return self.parse_node(value)
        elif isinstance(value, dict):
            dict_ = {}
            for key, value in value.items():
                key = self.parse_node_arg_value(key)
                value = self.parse_node_arg_value(value)
                dict_[key] = value
            return dict_
        else:
            return value

    def is_gunpowder_node(self, node):
        # ArrayKey node
        if isinstance(node, str) and node.isupper():
            return True
        elif isinstance(node, str) and node[0] == "_":
            return True

        if not isinstance(node, dict):
            return False

        node_name = list(node.keys())[0]
        return self.is_gunpowder_node_name(node_name)

    def is_gunpowder_node_name(self, node_name):
        node_name = snake_case_to_camel_case(node_name)
        return hasattr(cgp, node_name) or hasattr(gp, node_name)

    def get_node(self, node_name):
        if hasattr(cgp, node_name):
            return getattr(cgp, node_name)
        else:
            return getattr(gp, node_name)
