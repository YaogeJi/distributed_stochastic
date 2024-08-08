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
        'batch_size': {'values': [4]},
        'num_dimensions':{'values':[20000]},
        'sparsity':{'values':[9]},
        'num_nodes': {'values':[5]},
        'probability': {'values':[0.1]},
        'connectivity': {'values':[0.872]},
        'gamma': {'values': [28,32,36,40,44]},
        'lambda_const': {'values': [0, 0.1,0.2,0.3]},
        'solver':{'values':['sdmgt']},
        'max_iter': {'values': [1000]}        
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
