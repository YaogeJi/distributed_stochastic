import numpy as np
import time
from utils import proj_l1ball as proj
from sklearn import linear_model
import scipy
import copy
import wandb


class OnlineSolver(object):
    def __init__(self, generator, network, gamma, lmda, args) -> None:
        assert generator.m == network.shape[0]                # specifically for online version
        self.generator = generator
        self.network = network
        self.gamma = gamma
        self.lmda = lmda
        self.args = args
        p = (2 * np.log(self.args.num_dimensions) + 1)/(2 * np.log(self.args.num_dimensions))
        self.computation = self.args.computation
        self.communication = self.args.communication
        self.radius = np.linalg.norm(self.generator.theta.squeeze(), ord=p) * self.args.radius_const
        # self.theta = np.zeros([self.generator.m] + list(self.generator.theta.shape))
        self.theta = self.args.init_theta * np.ones([self.generator.m] + list(self.generator.theta.shape))
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


class CTA(OnlineSolver):
    def fit(self):
        for i in range(int(len(self.generator)/self.computation)): # if self.computation=1, this ratio means the total batches being used
            ref_theta = self.communicate(self.theta)
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                _, gamma = self.step(batch, ref_theta)
            self.theta_sum += self.theta * gamma
            self.gamma_sum += gamma
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
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
    def fit(self):
        for i in range(int(len(self.generator)/self.computation)): # if self.computation=1, this ratio means the total batches being used
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                _, gamma = self.step(batch, self.theta)
            self.theta = self.communicate(self.theta)
            self.theta_sum += self.theta * gamma
            self.gamma_sum += gamma
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
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
    def __init__(self, generator, network, gamma, lmda, args, prox=0) -> None:
        super(DualAveraging, self).__init__(generator, network, gamma, lmda, args)
        self.prox = prox
        self.mu = np.zeros([self.generator.m] + list(self.generator.theta.shape))
        self.prox_center = self.prox * np.ones_like(self.theta)
        self.sample_count = 0

    def fit(self):
        for i in range(int(len(self.generator)/self.computation)):
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                n, gamma = self.step(batch)
                # self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
                self.theta_sum += self.theta * gamma
                self.gamma_sum += gamma
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
        return True

    def step(self, batch):
        x, y = batch
        m, n, d = x.shape
        self.sample_count += n
        gamma = self.gamma() # 0.0147
        # lmda = self.lmda()
        p = (2 * np.log(d) + 1)/(2 * np.log(d))
        q = p / (p - 1)
        self.iter += 1
        self.mu = self.communicate(self.mu)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        self.mu += gradient
        #self.mu += gradient + lmda * np.sign(self.theta)
        xi = np.clip((p /(2 * np.e * np.log(d)) * gamma * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) * self.radius) - 1, 0, None)
        print(xi)
        print("---------")
        #self.theta = np.expand_dims(self.prox_center.squeeze(axis=2) + ((self.radius)**2 * (p-1) * gamma)/(2) * np.abs(self.mu.squeeze(axis=2)) ** (q-1) * np.sign(self.mu.squeeze(axis=2)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q), axis=2)
        self.theta = np.expand_dims(self.prox_center.squeeze(axis=2) + ((self.radius)**2 *(p / (2 * np.e * np.log(d))) * gamma)/(1+xi) * np.abs(self.mu.squeeze(axis=2)) ** (q-1) * np.sign(-self.mu.squeeze(axis=2)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q), axis=2)
        #self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma


class DualAveragingATC(DualAveraging):
    def step(self, batch):
        x, y = batch
        m, n, d = x.shape
        self.sample_count += n
        gamma = self.gamma() # 0.0147
        # lmda = self.lmda()
        p = (2 * np.log(d) + 1)/(2 * np.log(d))
        q = p / (p - 1)
        self.iter += 1
        #lmda = self.args.lambda_const*self.generator.noise_dev * np.sqrt(np.log(d)/(n*self.iter))
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        self.mu += gradient
        self.mu = self.communicate(self.mu)
        #self.mu += gradient + lmda * np.sign(self.theta)
        xi = np.clip((p /(2 * np.e * np.log(d)) * gamma * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) * self.radius) - 1, 0, None)
        print(xi)
        print("---------")
        #self.theta = np.expand_dims(self.prox_center.squeeze(axis=2) + ((self.radius)**2 * (p-1) * gamma)/(2) * np.abs(self.mu.squeeze(axis=2)) ** (q-1) * np.sign(self.mu.squeeze(axis=2)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q), axis=2)
        self.theta = np.expand_dims(self.prox_center.squeeze(axis=2) + ((self.radius)**2 *(p / (2 * np.e * np.log(d))) * gamma)/(1+xi) * np.abs(self.mu.squeeze(axis=2)) ** (q-1) * np.sign(-self.mu.squeeze(axis=2)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q), axis=2)
        #self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma


class DualAveragingConstraint(DualAveraging):
    def __init__(self, generator, network, gamma, lmda, args, prox=0) -> None:
        super().__init__(generator, network, gamma, lmda, args, prox)

    def step(self, batch):
        x, y = batch
        m, n, d = x.shape
        self.sample_count += n
        gamma = self.gamma() # 0.0147
        lmda = self.lmda()
        p = (2 * np.log(d) + 1)/(2 * np.log(d))
        q = p / (p - 1)
        self.iter += 1
        
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        # self.mu += gradient
        self.mu += gradient + lmda * np.sign(self.theta)
        self.mu = self.communicate(self.mu)
        xi = np.clip((p /(2 * np.e * np.log(d)) * gamma * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) * self.radius) - 1, 0, None)

        self.theta = np.expand_dims(self.prox_center.squeeze(axis=2) + ((self.radius)**2 *(p / (2 * np.e * np.log(d))) * gamma)/(1+xi) * np.abs(self.mu.squeeze(axis=2)) ** (q-1) * np.sign(-self.mu.squeeze(axis=2)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True) ** (2 - q), axis=2)
        #self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma


class SMD_constrained(OnlineSolver):
    def __init__(self, generator, network, gamma, lmda, args, prox=0) -> None:
        super(SMD_constrained, self).__init__(generator, network, gamma, lmda, args)
        self.prox = prox
        self.prox_center = self.prox * np.ones_like(self.theta)
        self.sample_count = 0
    

    def fit(self):
        for i in range(int(len(self.generator)/self.computation)):
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                n, gamma = self.step(batch)
                # self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
                self.theta_sum += self.theta * gamma
                self.gamma_sum += gamma
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
        return True


    def step(self, batch):
        def norm(matrix, p):
            #print(matrix.squeeze(axis=2).shape)
            lp_norm = np.linalg.norm(matrix.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2-p)
            #print(lp_norm.shape)
            #mask = (lp_norm>=10**10)
            # print(matrix.squeeze(axis=2)[mask])
            #print("------------")
            #print(mask.shape)
            # lp_norm[mask] = 0
            return np.expand_dims(lp_norm * np.sign(matrix.squeeze(axis=2)) *  np.abs(matrix.squeeze(axis=2)) ** (p-1) , axis=2)   
        x, y = batch
        m, n, d = x.shape
        self.sample_count += n
        gamma = self.gamma()
        p = (2 * np.log(d) + 1)/(2 * np.log(d))
        q = p / (p - 1)
        self.iter += 1 
        const = 2 * np.e * np.log(d)  / (p * (self.radius)**2)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)
        #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        phi_gradient = const * norm(self.theta, p)
        phi_gradient = phi_gradient - gamma * gradient
        phi_gradient = self.communicate(phi_gradient)
        # print(phi_gradient.shape)
        xi = np.expand_dims(np.clip((1 / (self.radius * const) * np.linalg.norm(phi_gradient.squeeze(axis=2), ord=q, axis=1, keepdims=True)) - 1, 0, None), axis=2)
        # print(xi.shape)
        self.theta = self.prox_center + 1/(const * (1+xi)) * norm(phi_gradient, q)
        return n, gamma

class SIPDD_constrained(OnlineSolver):
    def __init__(self, generator, network, gamma, lmda, args, prox=0) -> None:
        super(SIPDD_constrained, self).__init__(generator, network, gamma, lmda, args)
        self.prox = prox
        self.nu_temp = np.zeros([self.generator.m] + list(self.generator.theta.shape))
        self.nu = np.zeros_like(self.nu_temp)
        self.prox_center = self.prox * np.ones_like(self.theta)
        self.sample_count = 0
        self.alpha = args.alpha

    def fit(self):
        for i in range(int(len(self.generator)/self.computation)):
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                n, gamma = self.step(batch)
                # self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
                self.theta_sum += self.theta * gamma
                self.gamma_sum += gamma
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
        return True

    def step(self, batch):
        def norm(matrix, p):
            lp_norm = np.linalg.norm(matrix.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2-p)
            return np.expand_dims(lp_norm * np.sign(matrix.squeeze(axis=2)) *  np.abs(matrix.squeeze(axis=2)) ** (p-1) , axis=2)   
        x, y = batch
        m, n, d = x.shape
        self.sample_count += n
        gamma = self.gamma()
        p = (2 * np.log(d) + 1)/(2 * np.log(d))
        q = p / (p - 1)
        self.iter += 1 
        # const = 2 * p * (self.radius)**2  / (np.e * np.log(d))
        const = 2 * np.e * np.log(d)  / (p * (self.radius)**2)
        phi_gradient = const * norm(self.theta, p)
        nu = (1-self.alpha) * self.nu_temp + self.alpha * phi_gradient
        nu =  self.communicate(nu)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        mu = nu - gamma * gradient
        xi = np.expand_dims(np.clip((1 / (self.radius * const) * np.linalg.norm(mu.squeeze(axis=2), ord=q, axis=1, keepdims=True)) - 1, 0, None), axis=2)
        # xi = np.clip((2 * p * self.radius/(np.e * np.log(d)) * np.linalg.norm(self.mu.squeeze(axis=2), ord=q, axis=1, keepdims=True)**(2-q+q/p)) - 1, 0, None)
        theta = self.prox_center + 1/(const * (1+xi)) * norm(mu, q)
        # print("Warning: Wrong If data covar matrix is not I.")
        # print("Here, check dimension and operation from this line!")
        d_f = 1/2 * np.sum((theta - self.theta) ** 2, axis=1).squeeze(axis=1)
        d_h = const / 2 * (np.linalg.norm(theta.squeeze(axis=2), axis=1, ord=p)**2 - np.linalg.norm(self.theta.squeeze(axis=2), axis=1, ord=p)**2) - np.sum(((theta-self.theta).squeeze(axis=2) * nu.squeeze(axis=2)), axis=1)
        assert d_f.shape == d_h.shape
        mask = (np.greater(d_h, gamma * d_f)).reshape(-1,1,1)
        # mask = np.zeros_like(mask)
        print(mask)
        self.nu = mask * nu + (1-mask) * self.nu_temp
        self.nu_temp = self.nu - gamma * gradient
        self.nu_temp =  self.communicate(self.nu_temp)
        xi = np.expand_dims(np.clip((1 / (self.radius * const) * np.linalg.norm(self.nu_temp.squeeze(axis=2), ord=q, axis=1, keepdims=True)) - 1, 0, None), axis=2)
        self.theta = self.prox_center + 1/(const * (1+xi)) * norm(self.nu_temp, q)
        return n, gamma


class DualAveragingWholeSpace(DualAveraging):
    def __init__(self, generator, network, gamma, lmda, args, prox=0) -> None:
        super().__init__(generator, network, gamma, lmda, args, prox)
        assert not np.any(self.mu)

    def step(self, batch):
        def norm(matrix, p):
            lp_norm = np.linalg.norm(matrix.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2-p)
            lp_norm[lp_norm>=10**10] = 0
            return np.expand_dims(lp_norm * np.sign(matrix.squeeze(axis=2)) *  np.abs(matrix.squeeze(axis=2)) ** (p-1) , axis=2)
        x, y = batch
        m, n, d = x.shape
        self.sample_count += n
        gamma = self.gamma()
        p = 1 / (np.log(d)) + 1
        q = 1 + np.log(d)
        self.iter += 1
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        self.mu -= gamma * gradient
        const = np.e * np.log(d) * d ** ((p - 1) * (2 - p) / p)
        self.theta = norm(self.mu, q) / const
        return n, gamma


class MirrorDescent_V0(DualAveraging):
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
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
        return True
    
    def step(self, batch):
        def norm(matrix, p):
            lp_norm = np.linalg.norm(matrix.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2-p)
            lp_norm[lp_norm>=10**10] = 0
            return np.expand_dims(lp_norm * np.sign(matrix.squeeze(axis=2)) *  np.abs(matrix.squeeze(axis=2)) ** (p-1) , axis=2)
        x, y = batch
        m, n, d = x.shape
        self.iter += 1
        self.sample_count += n
        gamma = self.gamma()
        p = 1 / (np.log(d)) + 1
        q = 1 + np.log(d)
        #self.theta = self.communicate(self.theta)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        const = np.e * np.log(d) * d ** ((p - 1) * (2 - p) / p)
        phi_gradient = const * norm(self.theta, p)
        phi_gradient = phi_gradient - gamma * gradient
        phi_gradient = self.communicate(phi_gradient)
        #nu = self.shrinkage(phi_gradient, gamma * lmda)
        self.theta = norm(phi_gradient, q) / const
        # self.theta = self.shrinkage(self.theta, lmda)
        # self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma


class MirrorDescent_CTA(DualAveraging):    
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
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
        return True
    
    def step(self, batch):
        def norm(matrix, p):
            lp_norm = np.linalg.norm(matrix.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2-p)
            lp_norm[lp_norm>=10**10] = 0
            return np.expand_dims(lp_norm * np.sign(matrix.squeeze(axis=2)) *  np.abs(matrix.squeeze(axis=2)) ** (p-1) , axis=2)
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
        const = np.e * np.log(d) * d ** ((p - 1) * (2 - p) / p)
        phi_gradient = const * norm(self.theta, p)
        phi_gradient = phi_gradient - gamma * gradient
        # phi_gradient = self.communicate(phi_gradient)
        nu = self.shrinkage(phi_gradient, gamma * lmda)
        self.theta = norm(nu, q) / const
        # self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma
    
class MirrorDescent_ATC_V1(DualAveraging):
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
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
        return True
    
    def step(self, batch):
        def norm(matrix, p):
            lp_norm = np.linalg.norm(matrix.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2-p)
            lp_norm[lp_norm>=10**10] = 0
            return np.expand_dims(lp_norm * np.sign(matrix.squeeze(axis=2)) *  np.abs(matrix.squeeze(axis=2)) ** (p-1) , axis=2)
        x, y = batch
        m, n, d = x.shape
        self.iter += 1
        lmda = self.args.lambda_const * self.generator.noise_dev * np.sqrt(np.log(d) / (n * self.iter))
        self.sample_count += n
        gamma = self.gamma()
        p = 1 / (np.log(d)) + 1
        q = 1 + np.log(d)
        #self.theta = self.communicate(self.theta)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        const = np.e * np.log(d) * d ** ((p - 1) * (2 - p) / p)
        phi_gradient = const * norm(self.theta, p)
        phi_gradient = phi_gradient - gamma * gradient
        phi_gradient = self.communicate(phi_gradient)
        nu = self.shrinkage(phi_gradient, gamma * lmda)
        self.theta = norm(nu, q) / const
        # self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma

class MirrorDescent_ATC_V2(DualAveraging):
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
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
        return True
    
    def step(self, batch):
        def norm(matrix, p):
            lp_norm = np.linalg.norm(matrix.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2-p)
            lp_norm[lp_norm>=10**10] = 0
            return np.expand_dims(lp_norm * np.sign(matrix.squeeze(axis=2)) *  np.abs(matrix.squeeze(axis=2)) ** (p-1) , axis=2)
        x, y = batch
        m, n, d = x.shape
        self.iter += 1
        lmda = self.args.lambda_const * self.generator.noise_dev * np.sqrt(np.log(d) / (n * self.iter))
        self.sample_count += n
        gamma = self.gamma()
        p = 1 / (np.log(d)) + 1
        q = 1 + np.log(d)
        #self.theta = self.communicate(self.theta)
        gradient = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)  #if either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        const = np.e * np.log(d) * d ** ((p - 1) * (2 - p) / p)
        phi_gradient = const * norm(self.theta, p)
        phi_gradient = phi_gradient - gamma * gradient
        #phi_gradient = self.communicate(phi_gradient)
        nu = self.shrinkage(phi_gradient, gamma * lmda)
        nu = self.communicate(nu)
        self.theta = norm(nu, q) / const
        # self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma



class SDMGT(DualAveraging):
    def __init__(self, generator, network, gamma, lmda, args):
        super(SDMGT, self).__init__(generator, network, gamma, lmda, args)
        if self.computation != 1:
            raise NotImplementedError("multi local updates not implemented")
        self.grad_track = np.zeros((self.generator.m, self.generator.dimension, 1))
        self.last_grad = np.zeros((self.generator.m, self.generator.dimension, 1))
    
    def fit(self):  
        for i in range(int(len(self.generator)/self.computation)):
            for local_rounds in range(self.computation):
                batch = self.generator.sample()
                n, gamma = self.step(batch)
                # self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
                self.theta_sum += self.theta * gamma
                self.gamma_sum += gamma
            iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            avg_iter_theta = iter_theta.mean(axis=0)
            repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            assert iter_theta.shape == repeat_ground_truth.shape
            loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss)}, step=i)
        return True

    def step(self, batch):
        def norm(matrix, p):
            lp_norm = np.linalg.norm(matrix.squeeze(axis=2), ord=p, axis=1, keepdims=True) ** (2-p)
            lp_norm[lp_norm>=10**10] = 0
            return np.expand_dims(lp_norm * np.sign(matrix.squeeze(axis=2)) *  np.abs(matrix.squeeze(axis=2)) ** (p-1) , axis=2)
        x, y = batch
        m, n, d = x.shape
        self.iter += 1
        self.sample_count += n
        p = 1 / (np.log(d)) + 1
        q = 1 + np.log(d)
        const = np.e * np.log(d) * d ** ((p - 1) * (2 - p) / p)
        gamma = self.gamma()
        lmda = self.args.lambda_const*self.generator.noise_dev * np.sqrt(np.log(d)/(n*self.iter))
        grad_now = 1 / n * x.transpose(0,2,1) @ (x @ self.theta - y)
        last_grad = self.last_grad
        self.last_grad = grad_now
        grad_track = self.grad_track
        grad_track = self.communicate(grad_track + grad_now - last_grad)
        self.grad_track = grad_track
        phi_gradient = const * norm(self.theta, p)
        phi_gradient = phi_gradient - gamma * grad_track
        nu = self.shrinkage(phi_gradient, gamma * lmda)
        nu = self.communicate(nu)
        self.theta = norm(nu, q) / const
        return n, gamma
