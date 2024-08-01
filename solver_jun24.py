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


# -------------------Below are in the Jan 2024 theoretical analysis -------------------#

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
        cumulative_norm_diff = 0.
        theta_sum = 0.
        gamma_sum = 0.
        cumulative_concensus_loss = 0.
        cumulative_dual_concensus_loss = 0.
        for i in range(int(len(self.generator))):
            # online sampling
            batch = self.generator.sample()
            # online gradient update
            n, gamma = self.step(batch)

            # logging and visualization
            ## iterate average
            theta = self.theta.squeeze(axis=2)       # m * d
            theta_sum += theta * gamma                # m * d
            gamma_sum += gamma                        # 1
            iter_theta = theta_sum / gamma_sum       # m * d
            avg_iter_theta = iter_theta.mean(axis=0, keepdims=True) # 1 * d
            fake_average_theta = self.fake_theta_avg  # 1 * d
            nu = self.nu.squeeze(axis=2)             # m * d

            # optimality gap
            function_loss = np.sum((iter_theta - self.generator.theta.T) ** 2) / self.generator.m           # if X covariance is not identity, this is wrong.
            loss = np.linalg.norm(iter_theta - self.generator.theta.T, ord='fro') ** 2 / (self.generator.m)
            l1_loss = np.linalg.norm(avg_iter_theta.squeeze() - self.generator.theta.squeeze(), ord=1) ** 2
            consensus_loss = (np.linalg.norm(theta - theta.mean(axis=0, keepdims=True), axis=1, ord=1) ** 2).mean()
            consensus_loss_2 = np.mean(np.sum(np.abs(theta - fake_average_theta), axis=1) ** 2)
            cumulative_concensus_loss += consensus_loss_2
            norm_diff = np.linalg.norm((fake_average_theta.squeeze(axis=0) - theta.mean(axis=0)), ord=1) ** 2
            cumulative_norm_diff += norm_diff
            dual_consensus_loss =  np.mean(np.max(np.abs(nu - nu.mean(axis=0, keepdims=True)), axis=1) ** 2)
            cumulative_dual_concensus_loss += dual_consensus_loss
            # self.theta_sum += self.theta * gamma
            # self.gamma_sum += gamma

            # iter_theta = self.theta_sum.squeeze(axis=2)/self.gamma_sum
            # avg_iter_theta = iter_theta.mean(axis=0)
            # repeat_ground_truth = np.repeat(self.generator.theta.T, self.generator.m, axis=0)
            # repeat_avg_iter_theta = np.repeat(np.expand_dims(avg_iter_theta, axis=0), self.generator.m, axis=0)
            # assert repeat_ground_truth.shape == repeat_avg_iter_theta.shape
            # assert iter_theta.shape == repeat_ground_truth.shape


            # loss = np.linalg.norm(iter_theta - repeat_ground_truth, ord='fro') ** 2 / (self.generator.m)
            # l1_loss = np.linalg.norm(avg_iter_theta-self.generator.theta.squeeze(), ord=1) ** 2
            # # l1_loss = np.linalg.norm(iter_theta-repeat_ground_truth, ord=1) ** 2 / (self.generator.m)
            # c_loss = np.mean(np.sum(np.abs(self.theta.squeeze(axis=2) - np.repeat((np.mean(self.theta.squeeze(axis=2), axis=0, keepdims=True)), self.generator.m, axis=0)), axis=1) ** 2)
            # diff = np.expand_dims(iter_theta, axis=2) - np.expand_dims(repeat_ground_truth, axis=2)
            # function_loss = np.mean((diff.transpose(0,2,1) @ diff).squeeze())
            # mismatch_term = (self.fake_theta_avg - self.generator.theta).T @ (self.theta.mean(axis=0) - self.fake_theta_avg)
            # another_term = np.mean(np.sum(np.abs(self.theta.mean(axis=0) - self.fake_theta_avg), axis=1) ** 2) / (2 * gamma)
            # repeat_avg_fake_theta = np.repeat(np.expand_dims(self.fake_theta_avg.squeeze(axis=1), axis=0), self.generator.m, axis=0)
            # another_con_loss = np.mean(np.sum(np.abs(self.theta.squeeze(axis=2) - repeat_avg_fake_theta), axis=1) ** 2)
            # self.mismatch += mismatch_term
            # self.another_term += another_term
            wandb.log({"iter_loss (log scale)": np.log(loss),"l1_loss (log scale)":np.log(l1_loss), "consensus_loss_1 (log scale):": np.log(consensus_loss), "function_gap (log scale):": np.log(function_loss), "cumulative_norm_diff": np.log(cumulative_norm_diff/i), "consensus_loss_2 (log scale)": np.log(consensus_loss_2), "cumulative_consensus_loss_2 (log scale)": np.log(cumulative_concensus_loss/i), "dual_consensus_loss": np.log(dual_consensus_loss),"cumulative_dual_consensus_loss (log scale)": np.log(cumulative_dual_concensus_loss/i)}, step=i)
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
        self.nu = nu
        self.theta = norm(nu, q) / const
        self.fake_theta_avg = (norm(nu.mean(axis=0, keepdims=True), q) / const).squeeze(axis=2)
        # self.theta = (proj(self.theta.squeeze(axis=2),  self.radius)).reshape(m, d, 1)
        return n, gamma


