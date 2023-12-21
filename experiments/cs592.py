import os


exp_name = "cs592"
batch_size_group = [5000, 500, 50, 10]
d = 10
s = 10
sigma = 0.1
m_group = [1, 10, 100, 500]
p_group = [1, 0.2, 0.05, 0.015]
rho_group = [0, 0.9, 0.9, 0.9]
gamma = 1
sigma = 0.1
max_iter = 100
solver = "atc"
seed = 8989
scheduler = "diminish"

for i, m in enumerate(m_group):
    p = p_group[i]
    rho = rho_group[i]
    batch_size = batch_size_group[i]
    params = "--batch_size {} -d {} -s {} --sigma {} -m {} -p {} -rho {} --max_iter {} --solver {}  --gamma {} --seed {} --scheduler {} --storing_filepath  ./output/{}/batch_size{}_d{}_s{}_sigma{}_seed{}/m{}_rho{}/ --storing_filename {}_gamma{}_{}.output".format(batch_size, d, s, sigma, m, p, rho, max_iter, solver, gamma, seed, scheduler, exp_name, batch_size, d, s, sigma, seed, m, rho, solver, gamma, scheduler)
    print('python main.py {}'.format(params))
    os.system('python main.py {}'.format(params))
