import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'multi',
    'entity': 'yaoji',
    'project': "distributed_stochastic",
    'program': 'main.py',
    'metric': {
        'goal': 'minimize', 
        'name': 'iter_loss (log scale)'
    },
    'parameters': {
        'batch_size': {'values': [1]},
        'connectivity': {'values':[0.872]},
        'factor': {'values': [0.5]},
        'factor_asyn': {'values': [0.3,0.5,0.7]},
        'gamma': {'values': [14]},
        'generator': {'values': ['gaussian']},
        'lmda': {'values': [0.005]},
        'num_asyn_stage': {'values': [20]},
        'num_dimensions':{'values':[20000]},
        'num_iter': {'values': [200]},
        'num_lincon_stage': {'values': [4]},
        'num_nodes': {'values':[5]},
        'radius': {'values': [350]},
        'sigma': {'values': [1]},
        'solver':{'values':['DSDA']},
        'sparsity':{'values':[9]},
        'seed': {'values': [1001]},
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
wandb.agent(sweep_id)

