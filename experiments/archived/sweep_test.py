import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'entity': 'yaoji',
    'project': "distributed_stochastic",
    'program': 'main.py',
    'metric': {
        'goal': 'minimize', 
        'name': 'iter_loss (log scale)'
    },
    'parameters': {
        'batch_size': {'values': [32]},
        'num_dimensions':{'values':[20000]},
        'sparsity':{'values':[9]},
        'num_nodes': {'values':[20]},
        'probability': {'values':[0.2]},
        'connectivity': {'values':[0.9]},
        'gamma': {'values':[0.01]},
        'solver':{'values':['netlasso','atc']},
        'max_iter': {'values': [1000]}        
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
