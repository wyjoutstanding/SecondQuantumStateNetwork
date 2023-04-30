import warnings
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler


class TrapezoidLR(_LRScheduler):
    # Warm up before 1/4 epochs and cool down after 3/4 epochs
    def __init__(self, optimizer, milestones, last_epoch=-1):
        self.milestones = Counter(milestones)
        super(TrapezoidLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return [self.piecewise(self.last_epoch, base_lr) for base_lr in self.base_lrs]

    def piecewise(self, x, lr):
        milestones = list(sorted(self.milestones.elements()))
        # start with 1
        x = x + 1
        if x <= milestones[0]:
            return lr/milestones[0]*x
        elif (x <= milestones[1]) and (x > milestones[0]):
            return lr
        elif (x <= milestones[2]) and (x > milestones[1]):
            return lr*(milestones[2]-x)/(milestones[2]-milestones[1])
        else:
            return 0


# https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/6b5e8953a80aef5b324104dc0c2e9b8c34d622bd/warmup_scheduler/scheduler.py#L5
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr for base_lr in self.base_lrs]
        return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def get_scheduler(sche_name, optimizer, learning_rate, epochs):
    if sche_name == 'decay':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs/7*3), int(epochs/7*5)], gamma=0.1)
        # warm up at the first 1/10 of total epochs
        scheduler = GradualWarmupScheduler(optimizer, int(epochs/10), scheduler)
    elif sche_name == 'cyclic':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=(learning_rate/100),
                                          max_lr=learning_rate, step_size_up=int(epochs*2/5),
                                          step_size_down=int(epochs*3/5), cycle_momentum=False)
    elif sche_name == 'trap':
        scheduler = TrapezoidLR(optimizer, milestones=[int(epochs/4), int(epochs*3/4), epochs])
    elif sche_name == 'const':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs/7*3), int(epochs/7*5)], gamma=1.0)
    else:
        raise "sche_name not recognized."
    return scheduler