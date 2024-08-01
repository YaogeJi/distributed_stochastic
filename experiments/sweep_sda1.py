import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_1',
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
        'gamma': {'values': [2,4,6,8,10]},
        'solver':{'values':['DSDA']},
        'num_iter': {'values': [100]},
        'num_lincon_stage': {'values': [4]},
        'num_asyn_stage': {'values': [2]},
        'sigma': {'values': [0.5]},
        'radius': {'values': [5,10,50]},
        'lmda': {'values': [5,10,50,100]},
        'factor': {'values': [0.5,0.6,0.7]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
wandb.agent(sweep_id)
