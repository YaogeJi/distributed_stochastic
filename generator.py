import numpy as np
from sklearn import linear_model


class Generator:
    def __init__(self, batch_size_generator, m, dimension, sparsity, noise_dev, seed) -> None:
        self.batch_size_generator = batch_size_generator
        self.m = m
        self.dimension = dimension
        print(self.dimension)
        self.sparsity = sparsity
        self.noise_dev = noise_dev # sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.theta = self._init_ground_truth()
        

    def _init_ground_truth(self):
        theta = self.rng.normal(1, 1, (self.dimension, 1))
        theta_abs = np.abs(theta)
        threshold = np.quantile(theta_abs, 1 - self.sparsity / self.dimension)
        mask = theta_abs > threshold
        theta = mask * theta
        return theta

    def sample(self):
        batch_size = self.batch_size_generator()
        X = self.rng.normal(0, 1, (self.m, batch_size, self.dimension))
        epsilon = self.rng.normal(0, self.noise_dev ** 2, (self.m, int(batch_size), 1))
        Y = X @ self.theta + epsilon
        return X, Y
        
    def __len__(self):
        return len(self.batch_size_generator)

class BoundedGenerator(Generator):    
    def __init__(self, batch_size_generator, m, dimension, B, sparsity, noise_dev, seed) -> None:
        super().__init__(batch_size_generator, m, dimension, sparsity, noise_dev, seed)
        self.B = B

    def sample(self):
        batch_size = self.batch_size_generator()
        X = self.rng.uniform(low=-self.B, high=self.B, size=(self.m, int(batch_size), self.dimension))
        epsilon = self.rng.normal(0, self.noise_dev ** 2, (self.m, int(batch_size), 1))
        Y = X @ self.theta + epsilon
        return X, Y
