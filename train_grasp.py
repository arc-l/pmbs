from dataset import GraspClassificationDataset
from vision.resnet import resnet18, resnet10
from vision.efficientnet import EfficientNet
import log_utils
import torch_utils

import argparse
import time
import datetime
import os

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


def parse_args():
    default_params = {
        "lr": 0.1,
        "batch_size": 256,
        "epochs": 90,  # CosineAnnealing, should end before warm start
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lr_step_size": 40,
        "lr_gamma": 0.1,
    }

    parser = argparse.ArgumentParser(description="Train lifelong")

    parser.add_argument(
        "--lr", action="store", type=float, default=default_params["lr"], help="Enter the learning rate",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        default=default_params["batch_size"],
        type=int,
        help="Enter the batchsize for training and testing",
    )
    parser.add_argument("--momentum", default=default_params["momentum"], type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=default_params["weight_decay"],
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--epochs", action="store", default=default_params["epochs"], type=int, help="Enter the epoch for training",
    )
    parser.add_argument("--dataset_root", action="store", help="Enter the path to the dataset")
    parser.add_argument("--pretrained_model", action="store", help="The path to the pretrained model")
    parser.add_argument(
        "--ratio", action="store", default=1, type=float, help="ratio of how many data we use",
    )
    parser.add_argument(
        "--lr-step-size", default=default_params["lr_step_size"], type=int, help="decrease lr every step-size epochs"
    )
    parser.add_argument(
        "--lr-gamma", default=default_params["lr_gamma"], type=float, help="decrease lr by a factor of lr-gamma"
    )
    parser.add_argument(
        "--test-only", dest="test_only", help="Only test the model", action="store_true",
    )

    args = parser.parse_args()

    return args


class GraspTrainer:
    def __init__(self, args):
        self.params = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "ratio": args.ratio,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
        }

        self.dataset_root = args.dataset_root
        self.pretrained_model = args.pretrained_model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        log_dir_path = self.dataset_root
        self.log_dir = os.path.join(log_dir_path, "runs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        timestamp_value = datetime.datetime.fromtimestamp(time.time())
        time_name = timestamp_value.strftime("%Y-%m-%d-%H-%M")
        self.log_dir = os.path.join(self.log_dir, time_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # self.tb_logger = SummaryWriter(self.log_dir)
        self.logger = log_utils.setup_logger(self.log_dir, "Grasp")
        self.test_only = args.test_only

    def main(self):
        # model = resnet10(pretrained=False, num_classes=1, input_channels=1)
        model = EfficientNet.from_name("efficientnet-b0", in_channels=1, num_classes=1)
        # model = EfficientNet.from_pretrained("efficientnet-b2", in_channels=1, num_classes=1, advprop=True)
        model = model.to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.params["lr"],
            momentum=self.params["momentum"],
            weight_decay=self.params["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.params["lr_step_size"], gamma=self.params["lr_gamma"]
        )
        start_epoch = 0

        if self.pretrained_model is not None:
            checkpoint = torch.load(self.pretrained_model)
            model.load_state_dict(checkpoint["model"], strict=False)
            # optimizer.load_state_dict(checkpoint["optimizer"])
            # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # start_epoch = checkpoint["epoch"] + 1
            # prev_params = checkpoint["params"]

        self.logger.info(f"Hyperparameters: {self.params}")
        if self.pretrained_model is not None:
            self.logger.info(f"Start from the pretrained model: {self.pretrained_model}")
            # self.logger.info(f"Previous Hyperparameters: {prev_params}")

        data_loader_train = self._get_data_loader(self.params["batch_size"], self.params["ratio"])
        data_loader_test = self._get_data_loader(self.params["batch_size"], self.params["ratio"], test=True)

        if self.test_only:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self._evaluate(model, criterion, data_loader_test)
            return

        for epoch in range(start_epoch, self.params["epochs"]):
            # warmup start
            if epoch < 0:
                warmup_factor = 0.001
                warmup_iters = min(1000, len(data_loader_train) - 1)
                current_lr_scheduler = torch_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
            else:
                current_lr_scheduler = lr_scheduler

            train_loss = self._train_one_epoch(
                model, criterion, optimizer, data_loader_train, current_lr_scheduler, epoch,
            )

            if epoch % 2 == 0 or (self.params["epochs"] - epoch) < 2:
                save_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "params": self.params,
                }
                torch.save(save_state, os.path.join(self.log_dir, f"grasp_model-{epoch}.pth"))
                self._evaluate(model, criterion, data_loader_test)

        #     self.tb_logger.add_scalars("Epoch_Loss", {"train": train_loss}, epoch)
        #     self.tb_logger.flush()

        # self.tb_logger.add_hparams(self.params, {"hparam/train": train_loss})
        self.logger.info("Training completed!")

    def _train_one_epoch(
        self, model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq=50,
    ):
        model.train()
        metric_logger = log_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", log_utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("img/s", log_utils.SmoothedValue(window_size=10, fmt="{value:.2f}"))

        header = f"Epoch: [{epoch}]"
        n_iter = 0
        losses = []
        total_iters = len(data_loader)

        for (images, labels) in metric_logger.log_every(data_loader, print_freq, self.logger, header):
            start_time = time.time()

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            output_prob = model(images)

            loss = criterion(output_prob, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            log_loss = loss.item()
            log_lr = optimizer.param_groups[0]["lr"]
            acc05, acc07 = log_utils.accuracy(output_prob, labels, probs=(0.5, 0.7))
            batch_size = images.shape[0]
            metric_logger.meters["acc05"].update(acc05.item(), n=batch_size)
            metric_logger.meters["acc07"].update(acc07.item(), n=batch_size)
            metric_logger.update(loss=log_loss, lr=log_lr)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
            # self.tb_logger.add_scalar("Step/Train/Loss", log_loss, total_iters * epoch + n_iter)
            # self.tb_logger.add_scalar("Step/LR", log_lr, total_iters * epoch + n_iter)
            losses.append(log_loss)

            if epoch == 0:
                lr_scheduler.step()
            n_iter += 1

        if epoch != 0:
            lr_scheduler.step(epoch)

        return sum(losses) / len(losses)

    def _evaluate(self, model, criterion, data_loader, print_freq=100, log_suffix=""):
        model.eval()
        metric_logger = log_utils.MetricLogger(delimiter="  ")
        header = f"Test: {log_suffix}"

        num_processed_samples = 0
        with torch.no_grad():
            for image, target in metric_logger.log_every(data_loader, print_freq, self.logger, header):
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = model(image)
                loss = criterion(output, target)

                acc03, acc05, acc07 = log_utils.accuracy(output, target, probs=(0.3, 0.5, 0.7))
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc03"].update(acc03.item(), n=batch_size)
                metric_logger.meters["acc05"].update(acc05.item(), n=batch_size)
                metric_logger.meters["acc07"].update(acc07.item(), n=batch_size)
                num_processed_samples += batch_size

        print(f"{header} Acc@0.3 {metric_logger.acc03.global_avg:.3f}")
        print(f"{header} Acc@0.5 {metric_logger.acc05.global_avg:.3f}")
        print(f"{header} Acc@0.7 {metric_logger.acc07.global_avg:.3f}")
        return metric_logger.acc05.global_avg, metric_logger.acc07.global_avg

    def _get_data_loader(self, batch_size, ratio=1, test=False):
        """Get data loader."""
        if not test:
            dataset = GraspClassificationDataset(os.path.join(self.dataset_root, "train"))
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            dataset = GraspClassificationDataset(os.path.join(self.dataset_root, "test"))
            sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True
        )

        return data_loader


if __name__ == "__main__":
    args = parse_args()
    trainer = GraspTrainer(args)
    trainer.main()
