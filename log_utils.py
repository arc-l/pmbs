from collections import defaultdict, deque
import datetime
import time
import logging
from termcolor import colored
import sys
import os
import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} (global_avg: {global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger(object):
    """https://github.com/pytorch/vision/blob/master/references/segmentation/utils.py"""

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, logger, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == (len(iterable) - 1):
                eta_seconds = iter_time.avg * (len(iterable) - 1 - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable) - 1,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            len(iterable) - 1,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {}".format(header, total_time_str))


class _ColorfulFormatter(logging.Formatter):
    """https://github.com/facebookresearch/detectron2/blob/299c4b0dbab6fe5fb81d3870636cfd86fc334447/detectron2/utils/logger.py"""

    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        elif record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "grey")
        else:
            return log
        return prefix + " " + log


def setup_logger(output_dir=None, name="Training"):
    """https://github.com/facebookresearch/detectron2/blob/299c4b0dbab6fe5fb81d3870636cfd86fc334447/detectron2/utils/logger.py"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    color_formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s", datefmt="%m/%d %H:%M:%S", root_name=name,
    )

    # stdout logging
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(color_formatter)
    logger.addHandler(ch)

    # file logging
    if output_dir is not None:
        filename = os.path.join(output_dir, "log.txt")

        fh = logging.FileHandler(filename)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#         if target.ndim == 2:
#             target = target.max(dim=1)[1]

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target[None])

#         res = []
#         for k in topk:
#             correct_k = correct[:k].flatten().sum(dtype=torch.float32)
#             res.append(correct_k * (100.0 / batch_size))
#         return res
def accuracy(output, target, probs=(0.5,)):
    with torch.no_grad():
        res = []
        pred = torch.sigmoid(output)
        batch_size = target.size(0)

        for prob in probs:
            correct = (pred > prob).float()
            correct = correct.eq(target)
            correct = correct.sum(dtype=torch.float32)
            res.append(correct * (100.0 / batch_size))

        return res



class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    