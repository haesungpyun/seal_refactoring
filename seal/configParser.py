import copy
import json
import logging
import os
import zlib
import cached_path as _cached_path


class ConfigParser:
    def __init__(
        self,
        path_to_config    
    ) -> None:
        self.config_path = path_to_config
    
    def get_config_as_dict(self):
        config_file = str(_cached_path.cached_path(self.config_path))

        try:
            from _jsonnet import evaluate_file
        except ImportError:
            def evaluate_file(filename: str, **_kwargs) -> str:
                with open(filename, "r") as evaluation_file:
                    return evaluation_file.read()
                
        config_dict = json.loads(evaluate_file(config_file))
        return config_dict
    
    