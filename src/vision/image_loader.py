"""
Script with Pytorch's dataloader class
"""

import glob
import os
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import csv
import pandas as pd


class ImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(
        self, class_labels: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        """Fetches all (image path,label) pairs in the dataset.

        Args:
            class_labels: the class labels dictionary, with keys being the classes in this dataset
        Returns:
            list[(filepath, int)]: a list of filepaths and their class indices
        """

        img_paths = []  # a list of (filename, class index)

        ############################################################################
        # Student code begin
        ############################################################################

        for class_name, label in class_labels.items():
            class_folder = os.path.join(self.curr_folder, class_name)
            for img_file in os.listdir(class_folder):
                if img_file.endswith(('.jpg')):  # filter for image files
                    img_path = os.path.join(class_folder, img_file)
                    img_paths.append((img_path, label))

        return img_paths
    
        raise NotImplementedError(
            "`load_imagepaths_with_labels` function in "
            + "`image_loader.py` needs to be implemented"
        )


        ############################################################################
        # Student code end
        ############################################################################

       

    def get_classes(self) -> Dict[str, int]:
        """Get the classes (which are folder names in self.curr_folder)

        NOTE: Please make sure that your classes are sorted in alphabetical order
        i.e. if your classes are ['apple', 'giraffe', 'elephant', 'cat'], the
        class labels dictionary should be:
        {"apple": 0, "cat": 1, "elephant": 2, "giraffe":3}

        If you fail to do so, you will most likely fail the accuracy
        tests on Gradescope

        Returns:
            Dict of class names (string) to integer labels
        """

        classes = dict()
        ############################################################################
        # Student code begin
        ############################################################################

        classes = {
                        "bedroom": 0,
                        "coast": 1,
                        "forest": 2,
                        "highway": 3,
                        "industrial": 4,
                        "insidecity": 5,
                        "kitchen": 6,
                        "livingroom": 7,
                        "mountain": 8,
                        "office": 9,
                        "opencountry": 10,
                        "store": 11,
                        "street": 12,
                        "suburb": 13,
                        "tallbuilding": 14
                    }


        return classes
        raise NotImplementedError(
            "`get_classes` function in "
            + "`image_loader.py` needs to be implemented"
        )

        ############################################################################
        # Student code end
        ############################################################################
        
    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None
        ############################################################################
        # Student code begin
        ############################################################################

        # Open the image at the specified path
        img = Image.open(path)
        
        # Convert the image to grayscale ('L' mode)
        img = img.convert("L")
        
        return img
        raise NotImplementedError(
            "`load_img_from_path` function in "
            + "`image_loader.py` needs to be implemented"
        )

        ############################################################################
        # Student code end
        ############################################################################
        

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """
        img = None
        class_idx = None

        ############################################################################
        # Student code start
        ############################################################################

        # Access index-th image in dataset, and retrieve its path and class_idx
        img_path, class_idx = self.dataset[index]
        img = self.load_img_from_path(img_path)
            
        # Apply the transform (if any) to the image
        if self.transform:
            img = self.transform(img)

        return img, class_idx
    
        raise NotImplementedError(
            "`__getitem__` function in "
            + "`image_loader.py` needs to be implemented"
        )

        ############################################################################
        # Student code end
        ############################################################################
        

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        l = 0

        ############################################################################
        # Student code start
        ############################################################################
        
        l = len(self.dataset)

        return l
        raise NotImplementedError(
            "`__len__` function in "
            + "`image_loader.py` needs to be implemented"
        )
        ############################################################################
        # Student code end
        ############################################################################
        
