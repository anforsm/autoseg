import gunpowder as gp
from copy import deepcopy


class PreprocessingPipeline:
    def __init__(self, normalize=None, augment=None, target=None, config=None):
        self.normalize = normalize
        self.augment = augment
        self.target = target
        self.config = config
        self.array_keys = set()

    @property
    def pipeline(self):
        pipeline = None
        for component in [self.normalize, self.augment, self.target]:
            if component is not None:
                if pipeline is None:
                    pipeline = component
                else:
                    pipeline += component
        return pipeline

    def __add__(self, other):
        return self.pipeline + other

    @staticmethod
    def snake_case_to_camel_case(snake_str):
        components = snake_str.split("_")
        return components[0].title() + "".join(x.title() for x in components[1:])

    def dict_to_node(self, node_name, kwargs, variables):
        node = getattr(gp, node_name)
        self.recursively_parse_string(kwargs, variables)

        try:
            return node(**kwargs)
        except TypeError as e:
            print(e)

    def pipeline_to_nodes(self, node_list, variables):
        pipeline = None
        for node_dict in node_list:
            node_name_sk, kwargs = list(node_dict.items())[0]
            node_name = PreprocessingPipeline.snake_case_to_camel_case(node_name_sk)
            node = self.dict_to_node(node_name, kwargs, variables)
            if pipeline is None:
                pipeline = node
            else:
                pipeline += node
        return pipeline

    def build_from_config(self, variables):
        if self.config is None:
            raise ValueError("No config provided")

        for component in ["normalize", "augment", "target"]:
            if component in self.config:
                val = self.pipeline_to_nodes(
                    node_list=self.config[component], variables=variables
                )
                setattr(self, component, val)

    def build_pipeline(self, variables={}):
        if self.pipeline is None:
            self.build_from_config(variables)
        return self.pipeline

    def parse_string(self, string, variables):
        if not isinstance(string, str):
            return string

        if all(c.isupper() or not c.isalpha() for c in string):
            ak = gp.ArrayKey(string)
            self.array_keys.add(ak)
            return ak
        elif string.startswith("$"):
            return eval(string[1:], deepcopy(variables))

        return string

    def recursively_parse_string(self, dict_, variables):
        for k, v in dict_.items():
            if isinstance(v, dict):
                self.recursively_parse_string(v, variables)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, str):
                        v[i] = self.parse_string(item, variables)
            elif isinstance(v, str):
                dict_[k] = self.parse_string(v, variables)
        return dict_
