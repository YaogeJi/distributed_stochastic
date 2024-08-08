import numpy as np
import os

exp_name = "test"
# batch_size_group = [5000, 500, 50, 10]
batch_size = 20
d_group = np.array([20000])
s_group = np.floor(np.log(d_group))
sigma = np.sqrt(0.5)
generator = "gaussian"
# m = 20
# p = 0.2
# rho = 0.9
m = 1
p = 1
rho = 0
# m_group = [1, 10, 100, 500]
# p_group = [1, 0.2, 0.05, 0.015]
# rho_group = [0, 0.9, 0.9, 0.9]
gamma = 31
scheduler = "const"
max_iter = 1000
#solver = "dual_average"
#solver = "atc"
solver = "dual_average_whole_space"
# solver = "mirror_descent_v0"
seed = 8989
total = 1


for i, d in enumerate(d_group):
    s = int(s_group[i])
    params = "--batch_size {} -d {} -s {} --sigma {} --generator {} -m {} -p {} -rho {} --max_iter {} --solver {}  --gamma {} --seed {} --scheduler {} --storing_filepath  ./output/{}/batch_size{}_d{}_s{}_sigma{}_generator{}_seed{}/m{}_rho{}/ --storing_filename {}_gamma{}_{}.output".format(batch_size, d, s, sigma, generator, m, p, rho, max_iter, solver, gamma, seed, scheduler, exp_name, batch_size, d, s, sigma, generator,  seed, m, rho, solver, gamma, scheduler)
    print('python main.py {}'.format(params))
    os.system('python main.py {}'.format(params))
