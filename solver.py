import torch
import time
from utils import proj_l1ball as proj
import copy
import wandb


class OnlineSolver(object):
    def __init__(self, generator, network, args) -> None:
        assert generator.m == network.shape[0]  # specifically for online version
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator = generator
        self.network = network.to(self.device)
        self.m = self.network.shape[0]
        self.dim = self.generator.theta.shape[0]
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lmda = args.lmda
        self.factor = args.factor
        self.factor_asyn = args.factor_asyn
        self.args = args
        self.radius = self.args.radius
        self.theta = self.args.init_theta * torch.ones([self.generator.m] + list(self.generator.theta.shape), device=self.device)
        self.theta_sum = torch.zeros_like(self.theta)
        self.gamma_sum = 0
        self.iter = 0

    def communicate(self, matrix):
        matrix = torch.matmul(torch.matrix_power(self.network, self.communication), matrix.squeeze(dim=2)).unsqueeze(dim=2)
        return matrix

    @staticmethod
    def shrinkage(x, regularization):
        return torch.sign(x) * torch.clamp(torch.abs(x) - regularization, min=0)


class DSDA(OnlineSolver):
    def __init__(self, generator, network, args):
        super(DSDA, self).__init__(generator, network, args)
        self.theta = torch.zeros((self.m, self.dim, 1), dtype=torch.float).to(self.device)
        self.theta_cent = self.theta.clone()
        self.nu = torch.zeros((self.m, self.dim, 1), device=self.device)
        self.p = 1 + 1 / torch.log(torch.tensor(self.dim, dtype=torch.float))
        self.q = 1 + torch.log(torch.tensor(self.dim, dtype=torch.float))
        self.stage = 1

    def nabla_psi_star(self, vec):
        return (torch.abs(vec) / (torch.exp(torch.tensor(1.0)) * torch.log(torch.tensor(self.dim, dtype=torch.float)))) ** (self.q - 1) * torch.sign(vec)

    def distance_generating(self, vec, norm):
        return self.radius ** 2 * self.bregman((vec - self.theta_cent) / self.radius, norm, dim=self.dim)

    def restart(self, iter_theta):
        self.theta_sum = torch.zeros_like(self.theta)
        self.nu = torch.zeros((self.m, self.dim, 1), device=self.device)
        self.gamma_sum = 0
        if self.stage == 1:
            self.radius = self.factor * self.radius
            self.lmda = self.factor * self.lmda
            self.theta_cent = iter_theta.clone()
        if self.stage != 1:
            self.radius = self.factor_asyn * self.radius
            self.lmda = self.factor_asyn * self.lmda
            self.theta_cent = iter_theta.clone()
            self.num_iter = int(self.num_iter / self.factor_asyn)
            self.gamma = self.factor_asyn * self.gamma

    def gradient(self, x, y):
        n = torch.tensor(x.shape[1], dtype=torch.float, device=self.device)
        return torch.matmul(x.transpose(1, 2), (torch.matmul(x, self.theta) - y)) / n

    def fit(self):
        for r in range(self.args.num_lincon_stage + self.args.num_asyn_stage):
            if r == self.args.num_lincon_stage - 1:
                self.stage = 2
            for i in range(self.num_iter):
                batch = self.generator.sample(self.batch_size)
                self.step(batch)
                if self.iter >= self.generator.dimension / self.generator.m / self.batch_size:
                    return True

                self.iter += 1
                self.theta_sum += self.theta * self.gamma
                self.gamma_sum += self.gamma
                iter_theta = self.theta_sum.squeeze(dim=2) / self.gamma_sum
                avg_iter_theta = iter_theta.mean(dim=0)
                repeat_ground_truth = self.generator.theta.T.repeat(self.generator.m, 1)
                assert iter_theta.shape == repeat_ground_truth.shape
                loss = torch.norm(iter_theta - repeat_ground_truth, p='fro') ** 2 / (self.generator.m)
                l1_loss = torch.norm(avg_iter_theta - self.generator.theta.squeeze(), p=1) ** 2
                # theta_norm = torch.norm(torch.mean(self.theta, dim=(0, 2)), p=1)
                if not self.args.no_wandb:
                    wandb.log({"iter_loss (log scale)": torch.log(loss), "l1_loss (log scale)": torch.log(l1_loss)}, step=self.iter)
                else:
                    print(f"iter_loss: {loss}, l1_loss: {l1_loss}, iter: {self.iter}")
            self.restart(iter_theta.unsqueeze(dim=2))
        return True

    def step(self, batch):
        # start = time.time()
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        n = x.shape[1]
        self.nu_update(x, y)
        subgradient = torch.sign(self.theta)
        # start_xi = time.time()
        term = (-self.radius * self.gamma * (self.nu + self.lmda * subgradient)).squeeze(dim=2)

        xi_right = 1e10 * torch.ones((self.m, 1), device=self.device)
        xi_left = torch.zeros((self.m, 1), device=self.device)
        crit = torch.ones((self.m, 1), device=self.device)
        mask_xi = torch.ones((self.m, 1), dtype=torch.bool, device=self.device)
        while torch.max(torch.abs(crit[mask_xi.squeeze(dim=1)])) > 1e-5:
            xi = (xi_left + xi_right) / 2
            crit = torch.sum(torch.clamp(self.nabla_psi_star((torch.abs(term) - xi) / self.radius ** 2), min=0), dim=1) - 1
            mask_crit = (crit > 0)
            xi_left[mask_crit] = xi[mask_crit]
            xi_right[~mask_crit] = xi[~mask_crit]
            mask_xi = xi > 1e-3
            if torch.sum(mask_xi) == 0:
                self.xi = torch.zeros((self.m, 1), device=self.device)
                break
            if torch.norm(xi_left - xi_right) < 1e-3:
                self.xi = xi * mask_xi
                break
        else:
            self.xi = xi * mask_xi
        wandb.log({"xi": torch.norm(self.xi)}, step=self.iter)

        # end_xi = time.time()
        mask_1 = (torch.abs(term) > self.xi)
        mask_2 = 2 * (term < 0) - 1
        theta_tilda = mask_1 * self.nabla_psi_star((term + (mask_2 * self.xi)) / (self.radius ** 2))
        self.theta = theta_tilda.unsqueeze(dim=2) * self.radius + self.theta_cent
        # end = time.time()
        # wandb.log({"root_finding_ratio": (end_xi - start_xi)/(end-start)}, step=self.iter)
    def nu_update(self, x, y):
        self.nu = (self.network @ (self.nu + self.gradient(x, y)).squeeze(dim=2)).unsqueeze(dim=2)


class DSMD(DSDA):
    def nabla_phi(self):
        vec = (self.theta - self.theta_cent) / self.radius
        return self.radius * (torch.exp(torch.tensor(1.0)) * torch.log(torch.tensor(self.dim, dtype=torch.float))) * torch.abs(vec) ** (self.p - 1) * torch.sign(vec)

    def nu_update(self, x, y):
        self.nu = torch.matmul(self.network, (self.gradient(x, y) - self.nabla_phi() / self.gamma).squeeze(dim=2)).unsqueeze(dim=2)


class DSGD(DSDA):
    def step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        m, n, d = x.shape
        self.theta = self.theta - self.gamma / n * torch.matmul(x.transpose(1, 2), (torch.matmul(x, self.theta) - y))
        self.theta = proj(self.theta.squeeze(dim=2), self.radius).reshape(m, d, 1).to(self.device)
        assert self.theta.shape == (m, d, 1)
        assert y.shape == (m, n, 1)
        assert x.shape == (m, n, d)


class DSGT(DSGD):
    def __init__(self, generator, network, args):
        super(DSGT, self).__init__(generator, network, args)
        if self.computation != 1:
            raise NotImplementedError("multi local updates not implemented")
        self.grad_track = torch.zeros((self.generator.m, self.generator.dimension, 1)).to(self.device)
        self.last_grad = torch.zeros((self.generator.m, self.generator.dimension, 1)).to(self.device)

    def step(self, batch, ref_theta):
        x, y = batch
        m, n, d = x.shape
        gamma = self.gamma()
        grad_now = 1 / n * x.transpose(0,2,1) @ (x @ ref_theta - y)
        last_grad = self.last_grad
        self.last_grad = grad_now
        grad_track = self.grad_track
        grad_track = torch.matrix_power(self.network, self.communication) @ (grad_track + grad_now - last_grad).squeeze(axis=2)
        grad_track = grad_track.unsqueeze(axis=2)
        self.theta -= gamma * grad_track
        self.theta = (proj(self.theta.squeeze(axis=2), self.radius)).reshape(m, d, 1)
        self.grad_track = grad_track
        return n, gamma