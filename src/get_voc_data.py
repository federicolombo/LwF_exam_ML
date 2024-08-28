import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
import torch 

import numpy as np 

from pathlib import Path 
from typing import Literal
import os 

class VOC_Dataset(datasets.VOCDetection):
    def __init__(self,
                 root: Path, 
                 year: str, 
                 image_set: str,
                 download: bool,
                 transform: torchvision.transforms, 
                 label_group: list[str]):
        
        super().__init__(
            root=root, 
            year=year,
            image_set=image_set,
            download=download,
            transform=transform
            )
        
        self.label_group = label_group
        
        
    def __getitem__(self, index: int):
        """
        Override __getitem__ to filter images based on labels in label_group.
        
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the binary label vector for the selected label group.
        """    
        while True:
            image, target = super().__getitem__(index)
            
            if self.label_group:
                result = np.zeros(len(self.label_group), dtype=np.float32)
                objects = target['annotation']['object']
                label_found = False

                if not isinstance(objects, list):
                    objects = [objects]  # Ensure it's a list
                
                for obj in objects:
                    if int(obj["difficult"]) == 0:
                        if obj["name"] in self.label_group:
                            result[self.label_group.index(obj["name"])] = 1
                            label_found = True

                if label_found:
                    target = torch.from_numpy(result)
                    return image, target
            else:
                # If no label group is specified, return the original image and target
                return image, target
            
            # Move to the next image if no label matches
            index = (index + 1) % len(self.images)
    
    def __len__(self):
        
        return len(self.images)
    
    @staticmethod
    def get_categories(path_to_Main: Path) -> list[str]:
        """ 
        Extract the categories from the VOC dataset.
        In particular from the VOC2012/ImageSets/Main path.
        
        Args:
            main_path (Path): The path to the Main directory.
            
        Returns:
            A list of the categories.
        """
        categories = []
        
        if not os.path.isdir(path_to_Main):
            raise FileNotFoundError(f"The directory {path_to_Main} does not exist.")
        else:
            for file in os.listdir(path_to_Main):
                if file.endswith("_train.txt"):
                    categories.append(file.split("_")[0])
                    
            return categories

def voc_dataloader(data_path: Path,
                       custom_transform: torchvision.transforms,
                       batch_size: int,
                       image_set: Literal["train", "val"],
                       label_group: list[str],
                       year: str = "2012",
                       download: bool = False,
                       num_workers: int = 0) -> torch.utils.data.DataLoader:
    """ 
    Create test and train dataloaders.
    
    Returns:
        A torch.utils.data.DataLoader instance of train or test datalaoder. 
    """
    
    custom_dataloader = VOC_Dataset(
        root=data_path,
        year=year,
        image_set=image_set,
        download=download,
        transform=custom_transform,
        label_group=label_group
    )

    
    custom_dataloader = DataLoader(custom_dataloader,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
    
    return custom_dataloader