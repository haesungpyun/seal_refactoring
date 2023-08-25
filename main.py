import sys, os
import json
import pkgutil
import importlib
import torch
import importlib
from seal.set_experiment import Experiment
from allennlp.common.util import import_module_and_submodules
import pathlib

def main(path_to_config:str = None):

    with open(path_to_config, 'r') as f:
        config = json.load(f)

    # package = importlib.import_module('seal')
    # path = getattr(package, "__path__", [])
    # for _, name, ispkg in pkgutil.walk_packages(path):
    #     importlib.import_module('seal'+'.'+name)
    
    if 'allennlp' in str(config):
        import_module_and_submodules('allennlp')
        import_module_and_submodules('allennlp_models')

    package = importlib.import_module('seal')
    path = getattr(package, "__path__", [])
    for path, subdirs, subfiles in os.walk('seal'):
        if "__pycache__" in path:
            continue
        path = pathlib.Path(path)
        path = '.'.join(path.parts)
        for subf in subfiles:
            if subf.startswith('.') or subf.endswith('.pyc'):
                continue
            if not subf.endswith('.py'):
                continue
            path = pathlib.Path(path)
            file_path = path.joinpath(subf) 
            dot_not = '.'.join(path.joinpath(subf.replace('.py','')).parts)
            try:    
                importlib.import_module(dot_not)
            except:
                continue
            # spec = importlib.util.spec_from_file_location(dot_not, file_path)
            # module = importlib.util.module_from_spec(spec)
            # sys.modules[dot_not] = module
    
    experiment = Experiment(config)
    # experiment.make_data_loader()
    train_dataloader, model, optimizers = experiment.set_experiment()

    trainer_config = config.pop('trainer')
    inner_mode = trainer_config['inner_mode']
    num_inner_step = trainer_config['num_steps'].pop(inner_mode, 0)

    outer_mode, num_outer_step = trainer_config['num_steps'].popitem()

    for epoch in range(trainer_config['epoch']):
        running_loss = 0.0
        running_accuracy = 0.0
        total_batch = len(train_dataloader)
        
        for i, (Xs, y) in enumerate(train_dataloader):
            for _ in range(num_outer_step):
                for _ in range(num_inner_step):
                    optimizers[inner_mode].zero_grad()

                    result = model(Xs, y, inner_mode)
                    loss = result['loss']
                    
                    loss.backward()
                    optimizers[inner_mode].step()
                
                optimizers[outer_mode].zero_grad()
                
                result = model(Xs, y, outer_mode)
                pred = result['y_pred']
                loss = result['loss']
                
                loss.backward()
                optimizers[outer_mode].step()
                
            running_loss += loss.item()
            running_accuracy += (torch.sum(pred == y)/len(train_dataloader))

        # print loss for every 100 epoch
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:>4d}\tLoss: {running_loss / total_batch:.5f}')
            print(f'Epoch: {epoch:>4d}\tAccuracy: {running_accuracy / total_batch:.5f}')
        if i == 20:
            break
            
    print("running loss:", running_loss / total_batch)
    print(f'--------------------- Epoch {epoch} ended ---------------------')
            
if __name__ == '__main__':
    main("./config.json")
