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
        'probability': {'values':[1]},
        'connectivity': {'values':[0]},
        'gamma': {'values': [15,20,25, 30]},
        'solver':{'values':['dual_average_whole_space', 'mirror_descent_v0']},
        'max_iter': {'values': [1000]}        
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
