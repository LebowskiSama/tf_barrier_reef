import albumentations.pytorch.transforms
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import albumentations as A
from pathlib import Path
from skimage import io
import ast
import typing
from utils import (iou_width_height as iou)
import config
from bbox_utils import BoundingBox


class TorchDataset(Dataset):
    """Custom Torch Dataset"""
    def __init__(self, dataframe, anchors=config.ANCHORS, S=config.S, transforms=True):
        """
        Params: 
        `dataframe` of image path and annotations
        `transforms` where applicable
        """
        self.df = dataframe
        self.transform = transforms
        # Grid Sizes
        self.S = S
        # All 3 scales put together
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        self.albumentations = A.Compose([
            A.LongestMaxSize(max_size=int(config.IMAGE_SIZE)),
            A.PadIfNeeded(
                min_height=int(config.IMAGE_SIZE),
                min_width=int(config.IMAGE_SIZE),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.HorizontalFlip(),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
            albumentations.pytorch.transforms.ToTensorV2(),
        ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],))

    @staticmethod
    def albu_yolo(box):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        try:
            bbox_albu = A.convert_bbox_to_albumentations((x1, y1, x2, y2), source_format="pascal_voc", rows=720, cols=1280)
            bbox_yolo = A.convert_bbox_from_albumentations(bbox_albu, target_format="yolo", rows=720, cols=1280, check_validity=True)
            return bbox_yolo
        except:
            return -1

    def __getitem__(self, idx):
        """
        Return a single sample by given idx (index)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Image Parsing
        img_path = Path(self.df.iloc[idx]["path"]).resolve()
        image = io.imread(img_path)

        # Parse box
        boxes_obj = np.array(ast.literal_eval(self.df.iloc[idx]["annotations"]))
        boxes = []
        for _, box in enumerate(boxes_obj):
            box = list(box.values())
            yolo_box = (self.albu_yolo(box))
            if type(yolo_box) == tuple:
                boxes.append(yolo_box)
            else:
                continue

        # Apply transforms as necessary
        if self.transform:
            transformed = self.albumentations(image=image, bboxes=boxes)
            image = transformed['image']
            boxes = transformed['bboxes']

        # Yolo related ops to assign right anchors
        # [p_o, x, y, w, h]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 5)) for S in self.S]
        # print([target.shape for target in targets])

        for box in boxes:
            # Add presence of object
            # np.insert(box, 0, 1, axis=0)
            # Calculate iou score for each box with anchor to assign best anchor
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            # Sort for best anchor
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                # 0, 1, 2 for seeing which scale is best
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode="trunc")
                # 0, 1, 2 depending on which scale we're going for
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # Choose the right scale
                S = self.S[scale_idx]

                # Assign bounding box to relative cell in grid
                # suppose x = 0.5, S=13 -> int(6) = 6 almost the center of the image
                i, j = int(S * y), int(S * x)
                # Select the proper anchor to look at
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # Make sure the anchor isn't taken and there isn't also another anchor in the
                # same scale for the same bounding box
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    w_cell, h_cell = width * S, height * S

                    box_coordinates = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                    # print(box_coordinates)
                    # Fill in targets
                    # Class label
                    # targets[scale_idx][anchor_on_scale, i, j, 0] = int(class_label)
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    has_anchor[scale_idx] = True

                # Ignore by thresh
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # -1 to indicate we ignore this prediction
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)

    def __len__(self):
        return len(self.df)


def make_train_val_loaders(path_to_df: str = Path("../input/train_folds.csv").resolve(), train_split: float = 0.8, shuffle: bool = False, **kwargs) -> typing.Tuple[DataLoader, DataLoader]:
    """Create and return the COTS Dataloader"""
    df = pd.read_csv(path_to_df)
    
    # transformed_dataset = TorchDataset(dataframe=df, transform=torchvision.transforms.Compose([transform_train]))
    transformed_dataset = TorchDataset(dataframe=df)
    
    # Create Train Test Dataloader Split
    train_split = train_split
    dataset_size = len(transformed_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))

    if shuffle:
        np.random.seed(47)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[:split], indices[split:]
    
    train_loader = DataLoader(transformed_dataset, sampler=SequentialSampler(train_indices), **kwargs)
    val_loader = DataLoader(transformed_dataset, sampler=SequentialSampler(train_indices), **kwargs)

    return train_loader, val_loader


# Sanity Check
# if __name__ == "__main__":
#
#     # Check for float64 precision retention
#     torch.set_printoptions(precision=8)
#
#     # self, dataframe, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None
#     train_dl, _ = make_train_val_loaders(batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,)
#     for idx, batch in enumerate(train_dl):
#         print(idx)
#         #
#         images, targets = batch
#         for target in targets:
#             obj = target[..., 0] == 1
#             print(target[..., 0:5][obj])
#
#         if idx == 300:
#             break
