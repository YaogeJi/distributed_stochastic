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
        'sparsity':{'values':[30]},
        'num_nodes': {'values':[5]},
        'probability': {'values':[0.1]},
        'connectivity': {'values':[0.873]},
        
        'init_theta': {'values': [0.5]},
        'solver':{'values':['DualAveragingConstraint']},
        'scheduler': {'values': ['multistage']},
        'sigma': {'values': [0.5]},
        'gamma_const': {'values': [0.1, 1, 10]},
        'radius_const': {'values':[1]},
        'lmda_const': {'values':[1]},
        'alpha': {'values': [0.99]},
        'max_iter': {'values': [1000]},
        "first_stage_num_iter": {'values': [10, 50, 100]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
wandb.agent(sweep_id)
