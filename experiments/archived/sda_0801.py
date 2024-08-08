import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'd20000_exp4',
    'entity': 'yaoji',
    'project': "distributed_stochastic",
    'program': 'main.py',
    'metric': {
        'goal': 'minimize', 
        'name': 'iter_loss (log scale)'
    },
    'parameters': {
        'batch_size': {'values': [1]},
        'num_dimensions':{'values':[20000]},
        'sparsity':{'values':[10]},
        'num_nodes': {'values':[5]},
        'connectivity': {'values':[0]},
        'gamma': {'values': [1,2,4,8]},
        'solver':{'values':['DSDA']},
        'num_iter': {'values': [2000]},
        'num_lincon_stage': {'values': [5]},
        'num_asyn_stage': {'values': [1]},
        'sigma': {'values': [0.5]},
        'radius': {'values': [50]},
        'lmda': {'values': [100, 200, 500]},
        'factor': {'values': [0.5]},
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
wandb.agent(sweep_id)
