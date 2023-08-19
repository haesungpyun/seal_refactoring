from collections import defaultdict
import difflib
import importlib
from typing import Any
import inspect


class Wrapper(object):
    def __init__(self):
        super().__init__()
        self.base_classes=None
        self.registerd_class=None

    def construct_module(
            self, 
            config : dict,
            class_name: str,
            **kwargs:Any
        ):

        module, constructor, args = self.import_module(config, class_name)
        # If you want to instantiate specific sub classes, instantiate some classes in here,
        # then pass them to self.check_module_args function as keyword argument format.
        # Or you can instantiate the object outside of this function, and simply pass them to 
        # this construct_module as kwargs. 
        checked_params = self.check_module_args(
            module=module, 
            constructor=constructor, 
            args=args, 
            **kwargs
        )
        
        if constructor is None:
            return module(**checked_params)
        else:
            return getattr(module, constructor)(**checked_params)
        
    
    def check_module_args(
        self,
        module: object,
        constructor: Any, 
        args: dict, 
        **kwargs:Any      
    ):
        if constructor is None :
            constructor = module
        else:
            try:
                constructor = getattr(module, constructor)
            except:
                constructor = getattr(module, constructor.__name__)

        checked_param = defaultdict()
        accecpt_kwargs = False

        # if "object_name" in str(args) or "module_dot_path" in str(args):
            # import 
        signature = inspect.signature(constructor)
        parameters = dict(signature.parameters)
        
        for param_name, param in parameters.items():
            if param_name == 'self':
                continue

            if param.kind == param.VAR_KEYWORD:
                accecpt_kwargs = True
                continue    
            
            if kwargs.get(param_name) is not None:
                checked_param[param_name] = kwargs.pop(param_name)
                continue
            
            try:
                config_param = args.pop(param_name)
            except:
                # Default value
                if param.default != inspect._empty:
                    checked_param[param_name] = param.default
                continue
        
            if type(config_param) in [str, int, bool, float, complex]:
                checked_param[param_name] = config_param
            else:
                checked_param[param_name] = self.construct_module(config_param, param_name)
        
        if accecpt_kwargs:
            checked_param.update(args)
            checked_param.update(kwargs)

        return checked_param
            
    def import_module(
            self,
            config : dict,
            class_name: str,
        ):
        try:
            module_dot_path = config['module_dot_path'] # torch.utils.data
        except:
            raise KeyError (
                    "'module_dot_path' must be given to import module.\
                    e.g DataLoader from torch: 'module_dot_path': 'torch.utils.data.DataLoader' \
                    e.g load_dataset from transformers: 'module_dot_path': 'transformers.load_dataset' \
                    If you want to use Allen NLP, \
                        'model_dot_path': 'allennlp' \
                        'model_dot_path': 'allennlp-model'"
                )
        
        module_args = defaultdict()
        module_args.update(config.get('args', {}))

        if 'allen' in module_dot_path:      
            try:
                object_name = config['object_name'] # DataLoader
            except:
                raise KeyError (
                    "'object_name'(string used when register the class)' is required for allennlp"
                    )
            
            if self.base_classes is None or self.registerd_class is None:
                from allennlp.common.registrable import Registrable
                self.registerd_class = Registrable._registry
                self.base_classes = {cls.__name__: cls for cls in self.registerd_class}

            try:
                class_name = difflib.get_close_matches(class_name, list(self.base_classes.keys()))[0]
                module, constructor = self.registerd_class[self.base_classes[class_name]][object_name]
            except:
                raise KeyError (f":{object_name} not registered in allennlp, allennlp_models, seal")

            from allennlp.common.params import Params
            module_args = Params(module_args)
            
            return module, constructor, module_args

        else:
            try:
                object_name = config['object_name'] # DataLoader
            except:
                raise KeyError (
                    f"Excepted 'object_name' to get requested class from {module_dot_path} \
                    (e.g DataLoader from torch: 'object_name': 'DataLoader' \
                    e.g BERTModel from transformers: 'object_name': 'BERTModel') "
                    )
                
            try:
                module = importlib.import_module(module_dot_path)   
            except:                
                if object_name == module_dot_path.split('.')[-1]:
                    module_dot_path = '.'.join(module_dot_path.split('.')[:-1])
                module = importlib.import_module(module_dot_path)
                
            object_to_call = getattr(module, object_name)
        
            if config.get('constructor'):
                return object_to_call, config.get('constructor'), module_args
            else:
                return object_to_call, None, module_args

