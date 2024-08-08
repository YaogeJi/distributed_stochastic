import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_4',
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
        'connectivity': {'values':[0.872]},
        'gamma': {'values': [8]},
        'solver':{'values':['DSDA']},
        'num_iter': {'values': [25]},
        'num_lincon_stage': {'values': [26]},
        'num_asyn_stage': {'values': [3]},
        'sigma': {'values': [0.5]},
        'radius': {'values': [5]},
        'lmda': {'values': [100]},
        'factor': {'values': [0.6]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
wandb.agent(sweep_id)
