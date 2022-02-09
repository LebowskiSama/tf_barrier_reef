import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import albumentations as A
from pathlib import Path
from skimage import io
import ast
import typing
from utils import (iou_width_height as iou)
import config


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
            # A.LongestMaxSize(max_size=int(config.IMAGE_SIZE )),
            # A.PadIfNeeded(
            #     min_height=int(config.IMAGE_SIZE),
            #     min_width=int(config.IMAGE_SIZE),
            #     border_mode=cv2.BORDER_CONSTANT,
            # ),
            A.RandomCrop(width=config.IMAGE_SIZE, height=config.IMAGE_SIZE)
        ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],))

    def to_yolo_boxes(self, box):
        """Return normalized coordinates, yolo format"""
        imh, imw = 720, 1280

        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        xc = ((x2 + x1) / 2) / imw
        yc = ((y2 + y1) / 2) / imh
        w = w / imw
        h = h / imh

        assert(x1 < imw)
        assert(x2 < imw)
        assert(y1 < imh)
        assert(y1 < imh)

        return np.array([xc, yc, w, h], dtype=np.float64)

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
            boxes.append(self.to_yolo_boxes(box))

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
                scale_idx = anchor_idx // self.num_anchors_per_scale
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

                    # Fill in targets
                    # Class label
                    # targets[scale_idx][anchor_on_scale, i, j, 0] = int(class_label)
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    has_anchor[scale_idx] = True

                # Ignore by thresh
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # -1 to indicate we ignore this prediction

        return image, targets

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
if __name__ == "__main__":
    
    # Check for float64 precision retention
    torch.set_printoptions(precision=8)
        
    # self, dataframe, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None
    train_dl, _ = make_train_val_loaders(batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,)
    for idx, batch in enumerate(train_dl):
        print(idx)
        if idx == 100:
            
            images, targets = batch
            break
