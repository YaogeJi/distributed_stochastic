import numpy as np

class Scheduler(object):
    def __init__(self, args):
        self.args = args
        self.start_stepsize = args.gamma
        self.count = 1
    def __call__(self, iteration):
        self.count += 1


class ConstScheduler(Scheduler):
    def __init__(self, args):
        super(ConstScheduler, self).__init__(args)

    def __call__(self, iteration=None):
        super(ConstScheduler, self).__call__(iteration)
        return self.start_stepsize


class DiminishScheduler(Scheduler):
    def __init__(self, args):
        super(DiminishScheduler, self).__init__(args)

    def __call__(self, iteration=None):
        if iteration is None:
            iteration = self.count
        gamma = self.start_stepsize / (2 + self.args.sigma * np.sqrt(iteration / self.args.num_nodes))
        super(DiminishScheduler, self).__call__(iteration)
        return gamma


class DiminishSchedulerSecond(Scheduler):
    def __init__(self, radius, args):
        super(DiminishSchedulerSecond, self).__init__(args)
        self.radius = radius

    def __call__(self, iteration=None):
        if iteration is None:
            iteration = self.count
        gamma = self.start_stepsize * self.radius / np.sqrt(iteration)
        super(DiminishSchedulerSecond, self).__call__(iteration)
        return gamma


if __name__ == "__main__":
    gamma_const = ConstScheduler(0.01)
    gamma_diminishing = DiminishScheduler()
    for it in range(100):
        print("This is iteration {}".format(it))
        print(gamma_const())
        print(gamma_diminishing())