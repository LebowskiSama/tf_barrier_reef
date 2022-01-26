from struct import pack
import pandas as pd
import numpy as np
import torch, torchvision
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from pathlib import Path
from skimage import io
from bbox_utils import BoundingBox
import ast
import typing
from utils import (iou_width_height as iou, non_max_suppression as nms)

from config import IMAGE_SIZE, BATCH_SIZE


class TorchDataset(Dataset):

    """Custom Torch Dataset"""
    def __init__(self, dataframe, image_size=416, S=[13, 26, 52], transform=None):
        """
        Params: 
        `dataframe` of image path and annotations
        `transforms` where applicable
        """
        self.df = dataframe
        self.transform = transform
        # self.S = S
        # self.ancchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # All 3 scales
        # self.num_anchors = self.anchors.shape[0]
        # self.num_anchors_per_scale = self.num_anchors // 3
        # self.C = C
        # self.ignore_iou_thresh = 0.5


    def __getitem__(self, idx):
        """
        Return a single sample by given idx (index)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ## Image Parsing
        # abs path for image sample
        img_path = Path(self.df.iloc[idx]["path"]).resolve()
        # Read image as numpy array
        img_array = io.imread(img_path)

        # Parse box string
        boxes_obj = np.array(ast.literal_eval(self.df.iloc[idx]["annotations"]))
        boxes = []
        for i, box in enumerate(boxes_obj):
            box = list(box.values())
            boxes.append(np.asarray(box))
        
        sample = (img_array, np.asarray(boxes, dtype=np.float64))

        # Apply transforms as necessary
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # Return total iterable dataset size
        return len(self.df)

class ToTensor:
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Permute image tensor dimensions to make them torch friendly"""
        image, boxes = sample  
        # H x W x C -> C x H x W
        image = image.transpose((2, 0, 1))
        # Return transformed image and leave boxes untouched
        return torch.from_numpy(image), torch.tensor(boxes, dtype=torch.float64)

class ToYoloAnchors():
    def to_yolo(self, x, y, w, h, img_dim) -> np.array:
        """Return normalized coordinates, yolo format"""
        imh, imw = img_dim
        x = x / imw
        y = y / imh
        w = w / imw
        h = h / imh

        # 1 prepended to indicate presence of COTS
        return np.array([1, x, y, w, h], dtype=np.float64)


    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Convert (x, y, w, h) anchors to Yolo Anchors"""
        image, boxes = sample
        
        # Check for presence of labels
        if len(boxes) == 0:
            return image, boxes
        else:            
            yolo_boxes = []
            for box in boxes:
                try:        
                    x, y, w, h = box
                    imw, imh = IMAGE_SIZE
                    assert(x < imw)
                    assert(y < imh)
                    assert(x + w < imw)
                    assert(y + h < imh)
                    # Convert to Yolo
                    box = self.to_yolo(x, y, w, h, IMAGE_SIZE)
                    yolo_boxes.append(box)

                except AssertionError:
                    # Wrong labels
                    pass

            return image, np.asarray(yolo_boxes, dtype=np.float64)
        

def custom_collate(batch):
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]

    return [images, boxes]


def make_train_val_loaders(path_to_df: str=Path("../input/train_folds.csv").resolve(), batch_size: int=16, train_split: float = 0.8, shuffle: bool=False, pin_memory: bool=False, prefetch_factor: int=2) -> typing.Tuple[DataLoader, DataLoader]:
    """Create and return the COTS Dataloader"""
    df = pd.read_csv(path_to_df)
    
    transformed_dataset = TorchDataset(dataframe=df, transform = torchvision.transforms.Compose([ToYoloAnchors(), ToTensor()]))
    
    # Create Train Test Dataloader Split
    train_split = train_split
    dataset_size = len(transformed_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    if shuffle:
        np.random.seed(47)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[:split], indices[split:]
    
    train_loader = DataLoader(transformed_dataset, collate_fn=custom_collate, batch_size=batch_size, sampler=SequentialSampler(train_indices), pin_memory=pin_memory, prefetch_factor=prefetch_factor, shuffle=shuffle)
    val_loader = DataLoader(transformed_dataset, collate_fn=custom_collate, batch_size=batch_size, sampler=SequentialSampler(val_indices), pin_memory=pin_memory, prefetch_factor=prefetch_factor, shuffle=shuffle)

    return train_loader, val_loader


# Sanity Check
if __name__ == "__main__":

    from pprint import pprint
    
    # Check for float64 precision retention
    torch.set_printoptions(precision=8)
        
    # self, dataframe, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None
    train_dl, _ = make_train_val_loaders()
    for idx, batch in enumerate(train_dl):
        print(idx)
        if idx == 3:
            
            images, boxes = batch
            pprint([box.shape for box in boxes])
            break
