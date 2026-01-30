# This document introduce how to build custom dataset

## Step 1

### Create Custom Dataset class

Open it `ultralytics/data/dataset.py` and write your custom dataset define code.

**It's notice that custom dataset class output should match YOLODataset of train task.**

example:

```python
class LabelmeDataset(YOLODataset):
    """labelme dataset."""
```

## Step 2

Open it `ultralytics/data/build.py` and route your custom dataset fmt.

example:

```python
def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
):
    """Build and return a YOLO dataset based on configuration parameters."""
    if "format" in data.keys():
        if data["format"] == "Labelme":
            dataset = LabelmeDataset
        else:
            raise ValueError(f"dataset type <{data['format']}> unsupported")
    else:
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
```

## Step 3

Write example dataset.yaml for showing how to use.

example:

```yaml
## Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Example usage: yolo train data=Labelme-seg.yaml

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
format: "Labelme" # dataset format
path: labelme # dataset root dir
train: train/images # train images (relative to 'path')
val: val/images # val images (relative to 'path')
test: # test images (optional)

# Classes
names: ["person", "cat", "dog", "..."]
```
