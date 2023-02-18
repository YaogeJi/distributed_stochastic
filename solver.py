import numpy as np
import time
from utils import proj_l1ball as proj
from sklearn import linear_model
import scipy
import copy
import wandb


class OnlineSolver(object):
    def __init__(self, generator, network, gamma, args) -> None:
        assert generator.m == network.shape[0]                # specifically for online version
        self.generator = generator
        self.network = network
        self.gamma = gamma
        self.args = args
        self.computation = self.args.computation
        self.communication = self.args.communication
        self.radius = np.linalg.norm(self.generator.theta, ord=1) * self.args.radius_const
        self.theta = np.zeros([self.generator.m] + list(self.generator.theta.shape))  # + means extending the list to a np tenser: m*d*1
        self.theta_sum = np.zeros_like(self.theta)
        self.gamma_sum = 0
        self.iter = 0

    def communicate(self, matrix):
        matrix = np.expand_dims(np.linalg.matrix_power(self.network, self.communication) @ matrix.squeeze(axis=2), axis=2)
        return matrix

    @staticmethod
    def shrinkage(x, regularization):
        return np.sign(x) * np.clip(np.abs(x)-regularization, 0, None)


class CTA(OnlineSolver):
    def __init__(self, generator, network, gamma, args) -> None:        
        super(CTA, self).__init__(generator, network, gamma, args)

    def fit(self):
        for i in range(int(len(self.generator)/self.computation)): # if self.computation=1, this ratio means the total batches being used
            ref_theta = self.communicate(self.theta)
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                _, gamma = self.step(batch, ref_theta)
            self.theta_sum += self.theta * gamma
            self.gamma_sum += gamma
            loss = np.linalg.norm(self.theta_sum.squeeze()/self.gamma_sum - np.repeat(self.generator.theta.T, self.generator.m, axis=0), ord=2) ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm((self.theta_sum.squeeze()/self.gamma_sum).mean(axis=0) - self.generator.theta.squeeze()) 
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":l1_loss}, step=i)
        return True

    def step(self, batch):
        x, y = batch
        m, n, d = x.shape
        gamma = self.gamma()
        self.theta = self.theta - gamma / n * x.transpose(0,2,1) @ (x @ self.theta - y)
        self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
        assert self.theta.shape == (m, d, 1)
        assert y.shape == (m, n, 1)
        assert x.shape == (m, n, d)
        return n, gamma


class ATC(OnlineSolver):
    def __init__(self, generator, network, gamma, args) -> None:
        super(ATC, self).__init__(generator, network, gamma, args)
    
    def fit(self):
        for i in range(int(len(self.generator)/self.computation)): # if self.computation=1, this ratio means the total batches being used
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                _, gamma = self.step(batch, self.theta)
            self.theta = self.communicate(self.theta)
            self.theta_sum += self.theta * gamma
            self.gamma_sum += gamma
            loss = np.linalg.norm(self.theta_sum.squeeze()/self.gamma_sum - np.repeat(self.generator.theta.T, self.generator.m, axis=0), ord=2) ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm((self.theta_sum.squeeze()/self.gamma_sum).mean(axis=0) - self.generator.theta.squeeze()) 
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":l1_loss}, step=i)
        return True


class NetLasso(CTA):
    def __init__(self, generator, network, gamma, args):
        super(NetLasso, self).__init__(generator, network, gamma, args)
        if self.computation != 1:
            raise NotImplementedError("multi local updates not implemented")
        self.grad_track = np.zeros((self.generator.m, self.generator.dimension, 1))
        self.last_grad = np.zeros((self.generator.m, self.generator.dimension, 1))
        self.cta = False
    
    def step(self, batch, ref_theta):
        x, y = batch
        m, n, d = x.shape
        gamma = self.gamma()
        grad_now = 1 / n * x.transpose(0,2,1) @ (x @ ref_theta - y)
        last_grad = self.last_grad
        self.last_grad = grad_now
        grad_track = self.grad_track
        grad_track = np.linalg.matrix_power(self.network, self.communication) @ (grad_track + grad_now - last_grad).squeeze(axis=2)
        grad_track = np.expand_dims(grad_track, axis=2)

        self.theta -= gamma * grad_track
        self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
        self.grad_track = grad_track
        return n, gamma


class DualAveraging(OnlineSolver):
    def __init__(self, generator, network, gamma, args, prox=0) -> None:
        super(DualAveraging, self).__init__(generator, network, gamma, args)
        self.prox = prox
        self.mu = np.zeros([self.generator.m] + list(self.generator.theta.shape))
        self.prox_center = self.prox * np.zeros_like(self.theta)
        self.sample_count = 0

    def fit(self):
        for i in range(int(len(self.generator)/self.computation)):
            self.mu = self.communicate(self.mu)
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                n, gamma = self.step(batch)
                # self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
                self.theta_sum += self.theta * gamma
                self.gamma_sum += gamma
            loss = np.linalg.norm(self.theta_sum.squeeze()/self.gamma_sum - np.repeat(self.generator.theta.T, self.generator.m, axis=0), ord=2) ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm((self.theta_sum.squeeze()/self.gamma_sum).mean(axis=0) - self.generator.theta.squeeze()) 
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":l1_loss}, step=i)
        return True

    def step(self, batch):
        x, y = batch
        m, n, d = x.shape
        self.sample_count += n
        gamma = self.gamma()
        p = (2 * np.log(d) + 1)/(2 * np.log(d))
        q = p / (p - 1)
        self.iter += 1
        lmda = self.args.lambda_const*self.generator.noise_dev * np.sqrt(np.log(d)/(n*self.iter))
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        self.mu += gradient + lmda * np.sign(self.theta)
        xi = np.clip((p/(2 * np.e * np.log(d)) * gamma * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True)**(2-q+q/p) * self.radius) - 1, 0, None)
        #self.theta = np.expand_dims(self.prox_center.squeeze(axis=2) + ((self.radius)**2 * (p-1) * gamma)/(2) * np.abs(self.mu.squeeze(axis=2)) ** (q-1) * np.sign(self.mu.squeeze(axis=2)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q), axis=2)
        self.theta = np.expand_dims(self.prox_center.squeeze(axis=2) + ((self.radius)**2 *(p / (2 * np.e * np.log(d))) * gamma)/(1+xi) * np.abs(self.mu.squeeze(axis=2)) ** (q-1) * np.sign(-self.mu.squeeze(axis=2)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q), axis=2)
        #self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma


class MirrorDescent(DualAveraging):
    def __init__(self, generator, network, gamma, args) -> None:
        super().__init__(generator, network, gamma, args)
        #self.cta = False
    
    def fit(self):  
        loss_matrix = []
        theta_matrix = []
        mu_matrix = []
        for i in range(int(len(self.generator)/self.computation)):
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                n, gamma = self.step(batch)
                # self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
                self.theta_sum += self.theta * gamma
                self.gamma_sum += gamma
            loss = np.linalg.norm(self.theta_sum.squeeze()/self.gamma_sum - np.repeat(self.generator.theta.T, self.generator.m, axis=0), ord=2) ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm((self.theta_sum.squeeze()/self.gamma_sum).mean(axis=0) - self.generator.theta.squeeze()) 
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":l1_loss}, step=i)
        return True
    
    def step(self, batch):
        x, y = batch
        m, n, d = x.shape
        self.iter += 1
        lmda = self.args.lambda_const * self.generator.noise_dev * np.sqrt(np.log(d) / (n * self.iter))
        self.sample_count += n
        gamma = self.gamma()
        p = 1 / (np.log(d)) + 1
        q = 1 + np.log(d)
        self.theta = self.communicate(self.theta)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        phi_gradient = np.e * np.log(d) * d ** ((p - 1) * (2 - p) / p) * np.expand_dims(np.linalg.norm(self.theta.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2 - p) * np.sign(self.theta.squeeze(axis=2)) *  np.abs(self.theta.squeeze(axis=2)) ** (p-1) , axis=2)
        #phi_gradient = np.e * np.log(d)  * np.expand_dims(np.linalg.norm(self.theta.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2 - p) * np.sign(self.theta.squeeze(axis=2)) *  np.abs(self.theta.squeeze(axis=2)) ** (p-1) , axis=2)
        phi_gradient = phi_gradient - gamma * gradient 
        #self.theta = self.communicate(self.theta)
        #phi_gradient = self.communicate(phi_gradient)
        nu = self.shrinkage(phi_gradient, gamma*lmda)
        self.theta = 1 / (np.e * np.log(d) * d ** ((p - 1) * (2 - p) / p)) * np.expand_dims(np.linalg.norm(nu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q) * np.sign(nu.squeeze(axis=2)) *  np.abs(nu.squeeze(axis=2)) ** (q-1) , axis=2)
        #self.theta = 1/( np.e * np.log(d)) * np.expand_dims(np.linalg.norm(nu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q) * np.sign(nu.squeeze(axis=2)) *  np.abs(nu.squeeze(axis=2)) ** (q-1) , axis=2)
        self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma


class MirrorDescent_project(DualAveraging):
    def __init__(self, generator, network, gamma, args) -> None:
        super().__init__(generator, network, gamma, args)
        #self.cta = False
    
    def fit(self):  
        for i in range(int(len(self.generator)/self.computation)):
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                n, gamma = self.step(batch)
                self.theta_sum += self.theta * gamma
                self.gamma_sum += gamma
            loss = np.linalg.norm(self.theta_sum.squeeze()/self.gamma_sum - np.repeat(self.generator.theta.T, self.generator.m, axis=0), ord=2) ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm((self.theta_sum.squeeze()/self.gamma_sum).mean(axis=0) - self.generator.theta.squeeze()) 
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":l1_loss}, step=i)
        return True
    
    def step(self, batch):
        x, y = batch
        m, n, d = x.shape
        self.iter += 1
        lmda = self.args.lambda_const*self.generator.noise_dev * np.sqrt(np.log(d)/(n*self.iter))
        self.sample_count += n
        gamma = self.gamma()
       # p = 1/(np.log(d))+1
        p=2
        q=2
        #q = 1 + np.log(d)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        phi_gradient = np.expand_dims(np.e * np.log(d) * np.sign(self.theta.squeeze(axis=2)) *  np.abs(self.theta.squeeze(axis=2)) ** (p-1) , axis=2)
       # phi_gradient = np.e * np.log(d)  * np.expand_dims(np.linalg.norm(self.theta.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2 - p) * np.sign(self.theta.squeeze(axis=2)) *  np.abs(self.theta.squeeze(axis=2)) ** (p-1) , axis=2)
        phi_gradient = phi_gradient - gamma * gradient 
        #self.theta = self.communicate(self.theta)
        phi_gradient = self.communicate(phi_gradient)
        nu = self.shrinkage(phi_gradient, gamma*lmda)
        self.theta = 1/( np.e * np.log(d) ) * np.expand_dims(np.sign(nu.squeeze(axis=2)) *  np.abs(nu.squeeze(axis=2)) ** (q-1) , axis=2)
       # self.theta = 1/( np.e * np.log(d)) * np.expand_dims(np.linalg.norm(nu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q) * np.sign(nu.squeeze(axis=2)) *  np.abs(nu.squeeze(axis=2)) ** (q-1) , axis=2)
        self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
        return n, gamma  
