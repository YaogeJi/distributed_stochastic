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
        'batch_size': {'values': [20]},
        'num_dimensions':{'values':[20000]},
        'sparsity':{'values':[9]},
        'num_nodes': {'values':[1]},
        'lambda_const': {'values':[0]},
        'probability': {'values':[1]},
        'connectivity': {'values':[0]},
        'alpha':{'values':[0.5]},
        'gamma': {'values': [45, 50, 60, 70, 80]},
        'solver':{'values':['DualAveraging', 'SIPDD_constrained']},
        'max_iter': {'values': [1000]}        
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
