import pandas as pd
import numpy as np
import torch, torchvision
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from skimage import io
import ast

class TorchDataset(Dataset):

    """Custom Torch Dataset"""
    def __init__(self, dataframe, transform=None):
        """
        Params: 
        `dataframe` of image path and annotations
        `transforms` where applicable
        """
        self.df = dataframe
        self.transform = transform

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

        ## Parsing Bounding Boxes
        # Important: 20 has been specified as max number of bounding boxes per sample
        # so as to avoid variable tensor shapes later on during the dataloading process
        # this acts like padding and keeps tensor shapes constant.
        # A more optimal way to pad is perhaps possible
        boxes = np.zeros(shape=(20, 4))
        # Parse box string
        boxes_obj = ast.literal_eval(self.df.iloc[idx]["annotations"])
        # Append as np.float32 array
        for i, box in enumerate(boxes_obj):
            box_as_list = np.array(list(box.values()))
            boxes[i] = box_as_list.astype("float32")
        
        sample = dict(image = img_array, boxes=np.array(boxes))

        # Apply transforms as necessary
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # Return total iterable dataset size
        return len(self.df)

class ToTensor(object):

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """Permute image tensor dimensions to make them torch friendly"""
        image, boxes = sample["image"], sample["boxes"]    
        # H x W x C -> C x H x W
        image = image.transpose((2, 0, 1))
        # Return transformed image and leave boxes untouched
        return {
            "image": torch.from_numpy(image),
            "boxes": torch.from_numpy(boxes)
        }

def TorchDataLoader(path_to_df: str="../input/train_folds.csv", batch_size: int=4, shuffle: bool=False, pin_memory: bool=False, prefetch_factor: int=2) -> DataLoader:
    """Create and return the COTS Dataloader"""
    df = pd.read_csv(path_to_df)
    # Instantiate Dataset
    # dataset = TorchDataset(dataframe=df) # Vanilla Dataset
    transformed_dataset = TorchDataset(dataframe=df, transform = torchvision.transforms.Compose([ToTensor()]))
    # Create dataloader
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, prefectch_factor=prefetch_factor)

    return dataloader