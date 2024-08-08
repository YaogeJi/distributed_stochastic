import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'sgd',
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
        'range': {'values': [1]},
        'gamma': {'values': [0.001]},
        'generator': {'values': ['gaussian']},
        'lmda': {'values': [20]},
        'num_asyn_stage': {'values': [5]},
        'num_dimensions':{'values':[20000]},
        'num_iter': {'values': [4000]},
        'num_lincon_stage': {'values': [8]},
        'num_nodes': {'values':[5]},
        'radius': {'values': [50]},
        'sigma': {'values': [1]},
        'solver':{'values':['DSGD']},
        'sparsity':{'values':[9]},
        'seed': {'values': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020]},
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_stochastic")
wandb.agent(sweep_id)
