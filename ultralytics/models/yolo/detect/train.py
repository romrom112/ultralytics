# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.aggregate_dataset import build_aggregate_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, data, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # it should also return the index of the dataset
        if isinstance(data, (list, tuple)):
            return build_aggregate_dataset(self.args, batch, data, mode=mode, rect=mode == "val", stride=gs)
        else:
            return build_yolo_dataset(self.args, batch, data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in ["train", "val"]
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        shuffle = mode == "train"

        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            if mode == "train":
                dataset = self.build_dataset(self.datasets, mode, batch_size)
                return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader
            else:
                res = []
                for dataset in self.datasets:
                    dataset = self.build_dataset(dataset, mode, batch_size)
                    dataloader = build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader
                    res.append(dataloader)
                return res




    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.names = self.datasets[0]["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        heads = {}
        for i, dataset in enumerate(self.datasets):
            head_name = dataset["head_name"]
            if head_name in heads:
                continue
            nc = dataset["nc"]
            model = DetectionModel(cfg, nc=nc, verbose=verbose and RANK == -1)
            if weights and not self.resume:
                model.load(weights)
            head = model.model[-1]  # Detect() module
            heads[dataset["head_name"]] = head

        model.set_heads(heads)
        if self.resume:
            model.load(weights)
        return model

    def get_validator(self, batch_size):
        """Returns a DetectionValidator for YOLO model validation."""
        head_names = [dataset["head_name"] for dataset in self.datasets]
        head_names = sorted(head_names)
        loss_names = "box_loss", "cls_loss", "dfl_loss"
        self.loss_names = [f"{head_name}_{loss_name}" for head_name in head_names for loss_name in loss_names]
        test_loaders = self.get_dataloader(batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
        )
        validators = [yolo.detect.DetectionValidator(
            test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        ) for test_loader in test_loaders]
        return validators

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%17s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        return
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
