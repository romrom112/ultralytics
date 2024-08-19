# Ultralytics YOLO 🚀, AGPL-3.0 licensefrom collections import Counter
from collections import Counter

from ultralytics.data import YOLODataset
from ultralytics.utils import colorstr


class DatasetAggregator(YOLODataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(num_of_samples for dataset, num_of_samples,_ in self.datasets)

    def __getitem__(self, idx):
        for dataset, max_samples_per_ds, head_name in self.datasets:
            if idx < max_samples_per_ds:
                sample = dataset[idx]
                sample['head_name'] = head_name

                return sample
            idx -= max_samples_per_ds
        raise IndexError(f"Index {idx} out of range for aggregated dataset")

    def build_transforms(self, hyp=None):
        for dataset, max_samples_per_ds, head_name in self.datasets:
            dataset.build_transforms(hyp)

    def print_dataset_info(self):
        # print the number of samples in each dataset, also fractions
        counter = Counter()
        for dataset, max_samples_per_ds, head_name in self.datasets:
            counter[dataset.data["dataset_name"]] = max_samples_per_ds
        print(f"Total {sum(counter.values())} ({', '.join(f'{k}: {v / sum(counter.values()) * 100:.2f}%' for k, v in counter.items())})")

    def shuffle(self):
        for dataset, _, _ in self.datasets:
            dataset.shuffle()


def build_aggregate_dataset(cfg, batch, datas, mode="train", rect=False, stride=32):
    datasets = []
    for data in datas:
        img_path = data["train"] if mode == "train" else data["val"]
        dataset = YOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
        samples_per_dataset = data.get("images_per_epoch", len(dataset))

        datasets.append((dataset, samples_per_dataset, data["head_name"]))
    agg = DatasetAggregator(datasets)
    agg.print_dataset_info()
    return agg
