import os
import numpy as np


exp_name = "vignette2_1"
# batch_size_group = [5000, 500, 50, 10]
batch_size = 32
d_group = np.array([3200, 1600, 800, 400])
s_group = np.floor(np.log(d_group))
sigma = np.sqrt(0.5)
m = 20
p = 0.2
rho = 0.9
# m_group = [1, 10, 100, 500]
# p_group = [1, 0.2, 0.05, 0.015]
# rho_group = [0, 0.9, 0.9, 0.9]
gamma = 0.01
scheduler = "const"
max_iter = 20000
solver = "atc"
seed = 8989
total = 1

for exp_num in range(total):
    for i, d in enumerate(d_group):
        s = int(s_group[i])
        params = "--batch_size {} -d {} -s {} --sigma {} -m {} -p {} -rho {} --max_iter {} --solver {}  --gamma {} --seed {} --scheduler {} --storing_filepath  ./output/{}/batch_size{}_d{}_s{}_sigma{}_seed{}/m{}_rho{}/ --storing_filename {}_gamma{}_{}.output".format(batch_size, d, s, sigma,  m, p, rho, max_iter, solver, gamma, seed, scheduler, exp_name, batch_size, d, s, sigma,  seed, m, rho, solver, gamma, scheduler)
        print('python main.py {}'.format(params))
        os.system('python main.py {}'.format(params))
    seed += 1
