import torch

class Generator:
    def __init__(self, m, dimension, sparsity, noise_dev, seed) -> None:
        self.m = m
        self.dimension = dimension
        print(self.dimension)
        self.sparsity = sparsity
        self.noise_dev = noise_dev  # sigma
        self.seed = seed
        torch.manual_seed(seed)
        self.theta = self._init_ground_truth()

    def _init_ground_truth(self):
        theta = torch.normal(0, 1, (self.dimension, 1), device='cuda')
        theta_abs = torch.abs(theta)
        threshold = torch.quantile(theta_abs, 1 - self.sparsity / self.dimension)
        mask = theta_abs > threshold
        theta = mask * theta
        return theta

    def sample(self, batch_size):
        X = torch.normal(0, 1, (self.m, batch_size, self.dimension), device='cuda')
        epsilon = torch.normal(0, self.noise_dev, (self.m, batch_size, 1), device='cuda')
        Y = torch.matmul(X, self.theta) + epsilon
        return X, Y


class BoundedGenerator(Generator):
    def __init__(self, m, dimension, B, sparsity, noise_dev, seed) -> None:
        super().__init__(m, dimension, sparsity, noise_dev, seed)
        self.B = B

    def sample(self, batch_size):
        X = torch.empty((self.m, batch_size, self.dimension), device='cuda').uniform_(-self.B, self.B)
        epsilon = torch.normal(0, self.noise_dev ** 2, (self.m, batch_size, 1), device='cuda')
        Y = torch.matmul(X, self.theta) + epsilon
        return X, Y
