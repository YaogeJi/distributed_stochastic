import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_finetune',
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
        'sparsity':{'values':[9]},
        'num_nodes': {'values':[5]},
        'connectivity': {'values':[0]},
        'gamma': {'values': [7,8,9,10]},
        'solver':{'values':['DSDA']},
        'num_iter': {'values': [500]},
        'num_lincon_stage': {'values': [8]},
        'num_asyn_stage': {'values': [0]},
        'sigma': {'values': [0.001]},
        'radius': {'values': [50, 100]},
        'lmda': {'values': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        'factor': {'values': [0.5, 0.6, 0.7, 0.8]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
wandb.agent(sweep_id)
