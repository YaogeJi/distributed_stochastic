import numpy as np
import time
from utils import proj_l1ball as proj
import copy
import wandb


class OnlineSolver(object):
    def __init__(self, generator, network, args) -> None:
        assert generator.m == network.shape[0]                # specifically for online version
        self.generator = generator
        self.network = network
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lmda = args.lmda
        self.factor = args.factor
        self.args = args
        self.radius = self.args.radius
        # self.theta = np.zeros([self.generator.m] + list(self.generator.theta.shape))
        self.theta = self.args.init_theta * np.ones([self.generator.m] + list(self.generator.theta.shape))
        # self.theta = np.random.normal(loc=0, scale=10, size=[self.generator.m] + list(self.generator.theta.shape))  # + means extending the list to a np tenser: m*d*1
        #self.theta = np.random.normal(loc=0, scale=0.5, size=[self.generator.m] + list(self.generator.theta.shape))  # + means extending the list to a np tenser: m*d*1
        self.theta_sum = np.zeros_like(self.theta)
        self.gamma_sum = 0
        self.iter = 0

    def communicate(self, matrix):
        matrix = np.expand_dims(np.linalg.matrix_power(self.network, self.communication) @ matrix.squeeze(axis=2), axis=2)
        return matrix

    @staticmethod
    def shrinkage(x, regularization):
        return np.sign(x) * np.clip(np.abs(x)-regularization, 0, None)


class DSDA(OnlineSolver):
    def __init__(self, generator, network, args):
        super(DSDA, self).__init__(generator, network, args)
        self.m = self.network.shape[0]
        self.dim = self.generator.theta.shape[0]
        self.theta = 0/self.m*np.ones((self.m, self.dim, 1))
        self.theta_cent = self.theta.copy()
        self.nu =  np.zeros((self.m, self.dim, 1))
        self.p = 1 + 1 / np.log(self.dim)
        self.q = 1 + np.log(self.dim)
        self.stage = 1

    def nabla_psi_star(self, vec):
        return (np.abs(vec) / (np.e * np.log(self.dim))) ** (self.q-1) * np.sign(vec) # done checking.
    
    def distance_generating(self, vec, norm):
        # Phi
        return self.radius ** 2 * self.bregman((vec - self.theta_cent) / self.radius, norm, dim=self.dim)
    
    def restart(self, iter_theta):
        self.radius = self.factor * self.radius
        self.lmda = self.factor * self.lmda
        self.theta_cent = iter_theta.copy()
        if self.stage != 1:
            self.num_iter = int(self.num_iter / self.factor)
            self.gamma = self.factor * self.gamma 
   
    def gradient(self, x, y):
        n = x.shape[1]
        return 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)
    
    def fit(self):
        for r in range(self.args.num_lincon_stage + self.args.num_asyn_stage):
            if r == self.args.num_lincon_stage - 1:
                self.stage = 2
            for i in range(self.num_iter):
                batch = self.generator.sample(self.batch_size)
                self.step(batch)
                self.iter += 1
                # self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
                self.theta_sum += self.theta * self.gamma
                self.gamma_sum += self.gamma
                iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
                avg_iter_theta = iter_theta.mean(axis=0)
                repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
                assert iter_theta.shape == repeat_ground_truth.shape
                loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
                l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
                theta_norm = np.linalg.norm(np.mean(self.theta, axis=(0,2)), ord=1)
                if not self.args.no_wandb: 
                    wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)": np.log(l1_loss), "theta_norm": theta_norm}, step=self.iter)
                else:
                    print(f"iter_loss: {loss}, l1_loss: {l1_loss}, iter: {self.iter}")
            self.restart(np.expand_dims(iter_theta, axis=2))
        return True

    def step(self, batch):
        x, y = batch
        n = x.shape[1]
        self.nu_update(x, y) # should we first update nu?
        # subgradient = np.sign(self.theta)
        subgradient = 2 * (self.theta >= 0) - 1
        term = (-self.radius * self.gamma * (self.nu + self.lmda * subgradient)).squeeze(axis=2) # this is a vector

        # first we calculate the new xi using bisection.

        xi_right = 1e10 * np.ones((self.m, 1))
        xi_left = np.zeros((self.m, 1))
        crit = np.ones((self.m, 1)) # some initial value that satisfy the while loop.
        mask_xi = np.ones((self.m, 1), dtype=bool)
        while np.max(np.abs(crit[mask_xi.squeeze(axis=1)])) > 1e-5:
            xi = (xi_left + xi_right) / 2
            crit = np.sum((np.clip(self.nabla_psi_star((np.abs(term) - xi)/self.radius**2), a_min=0, a_max=None)), axis=1) - 1 # m.
            mask_crit = (crit > 0)
            xi_left[mask_crit] = xi[mask_crit]
            xi_right[~mask_crit] = xi[~mask_crit]
            mask_xi = xi > 1e-5
            if np.sum(mask_xi) == 0:
                self.xi = np.zeros((self.m, 1))
                break
        else: # will not be execute if the loop is finished by break or raise.
            self.xi = xi * mask_xi
        wandb.log({"xi": np.linalg.norm(self.xi)}, step=self.iter)
        # elementwise calculate
        # mask_1: false means condition 1, true means condition 2 or 3.
        # mask_2: false means condition 3, true means condition 2.
        mask_1 = (np.abs(term) > self.xi) 
        mask_2 = 2 * (term < 0) - 1          # term<0, mask_2 = 1; term > 0, mask_2 = -1
        theta_tilda =  mask_1 * self.nabla_psi_star((term + (mask_2 * self.xi)) / (self.radius ** 2))   
        self.theta = np.expand_dims(theta_tilda, axis=2) * self.radius + self.theta_cent
    
    def nu_update(self, x, y):
        self.nu = np.expand_dims(self.network @ (self.nu + self.gradient(x, y)).squeeze(axis=2), axis=2)

class DSMD(DSDA):
    def nabla_phi(self):
        vec = (self.theta - self.theta_cent) / self.radius
        return self.radius * (np.e * np.log(self.dim)) * np.abs(vec) ** (self.p-1) * np.sign(vec) # 
    
    def nu_update(self, x, y):
        self.nu = np.expand_dims(self.network @ (self.gradient(x, y) - self.nabla_phi() / self.gamma).squeeze(axis=2), axis=2)

class DSGD(DSDA):
    # ATC version.
    def step(self, batch):
        x, y = batch
        m, n, d = x.shape
        self.theta = self.theta - self.gamma / n * x.transpose(0,2,1) @ (x @ self.theta - y)
        self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
        assert self.theta.shape == (m, d, 1)
        assert y.shape == (m, n, 1)
        assert x.shape == (m, n, d)
