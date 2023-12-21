import numpy as np

class Scheduler(object):
    def __init__(self, init_value, args):
        self.args = args
        self.init_value = init_value
        self.count = 1
    def __call__(self, iteration):
        self.count += 1


class ConstScheduler(Scheduler):
    def __init__(self, init_value, args):
        super(ConstScheduler, self).__init__(init_value, args)

    def __call__(self, iteration=None):
        super(ConstScheduler, self).__call__(iteration)
        return self.init_value


class DiminishScheduler(Scheduler):
    def __init__(self, init_value, args):
        super(DiminishScheduler, self).__init__(init_value, args)

    def __call__(self, iteration=None):
        if iteration is None:
            iteration = self.count
        value = self.init_value / (2 + self.args.sigma * np.sqrt(iteration / self.args.num_nodes))
        super(DiminishScheduler, self).__call__(iteration)
        return value


class DiminishSchedulerSecond(Scheduler):
    def __init__(self, init_value, radius, args):
        super(DiminishSchedulerSecond, self).__init__(init_value, args)
        self.radius = radius

    def __call__(self, iteration=None):
        if iteration is None:
            iteration = self.count
        value = self.init_value * self.radius / np.sqrt(iteration)
        super(DiminishSchedulerSecond, self).__call__(iteration)
        return value

class MultiStageScheduler(Scheduler):
    def __init__(self, init_value, first_stage_num_iter, multiplier, args):
        super(MultiStageScheduler, self).__init__(init_value, args)
        self.first_stage_num_iter = first_stage_num_iter
        self.multiplier = multiplier
        self.threshold =first_stage_num_iter
        self.value = init_value
        self.stage = 0  
        
    def __call__(self, iteration=None):
        if iteration is None:
            iteration = self.count
        if iteration == self.threshold:
            self.stage += 1
            self.threshold += self.first_stage_num_iter * self.multiplier ** self.stage
            self.value /= self.multiplier
        super(MultiStageScheduler, self).__call__(iteration)
        return self.value
 