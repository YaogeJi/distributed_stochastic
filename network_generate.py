import os
import pickle
from network import ErodoRenyi


num_nodes = 20
probability = 0.4
connectivity = 0.85

network_path = "./network/"
network_file = network_path + "m{}_rho{}.network".format(num_nodes, connectivity)

## processing network
try:
    w = pickle.load(open(network_file, "rb"))
except:
    w = ErodoRenyi(m=num_nodes, rho=connectivity, p=probability).generate()
    os.makedirs(network_path, exist_ok=True)
    pickle.dump(w, open(network_file, "wb"))
