import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'decentralized_mirror_descent_linear_speedup_beta',
    'entity': 'yaoji',
    'project': "distributed_stochastic",
    'program': 'main.py',
    'metric': {
        'goal': 'minimize', 
        'name': 'iter_loss (log scale)'
    },
    'parameters': {
        'batch_size': {'values': [1]},
        'num_dimensions':{'values': [100]},
        'sparsity':{'values':[10]},
        'num_nodes': {'values':[1000]},
        'connectivity': {'values':[0.8]},
        'gamma': {'values': [8,10]},
        "sigma": {'values': [1]},
        'lambda_const': {'values': [0]},
        'solver':{'values':['MirrorDescent_ATC_V1']},
        'max_iter': {'values': [200000]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")

wandb.agent(sweep_id)
