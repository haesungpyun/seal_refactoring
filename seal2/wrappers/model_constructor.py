from seal2.modules.sampler.sampler import AppendingSamplerContainer
from .constructor import Constructor

class ModelConstructor(Constructor): # type: ignore
    def __init__(self, config_dict):
        super().__init__()
        self.model_config = config_dict['model']
    
    def make_shared_nn_seal(self, vocab):
        # task_nn in config should be popped
        task_nn = self.construct_class(self.model_config['args'].pop('task_nn'), 'task_nn')
        score_nn = self.construct_class(self.model_config['args'].get('score_nn'), 'score_nn')
        oracle_value_function = self.construct_class(self.model_config['args'].get('oracle_value_function'), 'oracle_value_function')
        
        infnet_smapler = self.construct_class(
            self.model_config['args'].get('inference_module'), 
            'inference_module',
            inference_nn=task_nn,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
        )
        if  self.model_config['args'].get('sampler') is None:
                sampler_ = AppendingSamplerContainer(
                    score_nn=score_nn,
                    oracle_value_function=oracle_value_function,
                    constituent_samplers=[],
                    log_key="sampler",
                )
        else: 
            sampler_ = self.construct_class(
                self.model_config['args'].get('sampler'),
                'sampler',  
                score_nn=score_nn,
                oracle_value_function=oracle_value_function,
            )
        
        loss_fn = self.construct_class(
            self.model_config['args'].get('loss_fn'), 
            'loss_fn',
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
        )
        
        sampler_.append_sampler(infnet_smapler)
        
        inference_module = self.construct_class(
            self.model_config['args'].get('inference_module'), 
            'inference_module',
            inference_nn=task_nn,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
        )
              
        initializer = self.construct_class(
            self.model_config['args'].get('initializer'),
            "initializer"
        )

        regularizer = self.construct_class(
            self.model_config['args'].get('regularizer'),
            "regularizer"
        )

        seal = self.construct_class(
            self.model_config, 
            'model',
            sampler=sampler_,
            loss_fn=loss_fn,
            oracle_value_function=oracle_value_function,
            score_nn=score_nn,
            inference_module=inference_module,
            initializer=initializer,
            regularizer=regularizer
        )

        return seal

