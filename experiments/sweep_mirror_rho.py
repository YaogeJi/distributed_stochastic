import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'decentralized_mirror_descent_m20_step_size',
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
        'connectivity': {'values':[0, 0.5, 0.6, 0.7, 0.8, 0.873]},
        'gamma': {'values': [6,8,10,12, 14]},
        'lambda_const': {'values': [0.1]},
        'solver':{'values':['MirrorDescent_ATC_V1']},
        'max_iter': {'values': [4000]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")

wandb.agent(sweep_id)