from src import config
import torch
from torch.utils.data import Dataset
import cv2
import os


class ImageDataset(Dataset):
    # initialize the constructor
    def __init__(self, data, transforms=None):
        self.transforms = transforms
        self.data = data

    def __getitem__(self, index):
        # retrieve annotations from stored list
        filename, startX, startY, endX, endY, label = self.data[index]

        # get full path of filename
        image_path = os.path.join(config.IMAGES_PATH, label, filename)

        # load the image (in OpenCV format), and grab its dimensions
        image = cv2.imread(image_path)
        if image is None:
            return None, None, None
        h, w = image.shape[:2]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # scale bounding box coordinates relative to dimensions of input image
        # normalize bounding box coordinates in (0, 1)
        # x coordinates normalized by width, y coordinates normalized by height
        startX_norm = int(startX) / w
        startY_norm = int(startY) / h
        endX_norm = int(endX) / w
        endY_norm = int(endY) / h
        
        # create bbox tensor with normalized coordinates
        bbox = torch.tensor([startX_norm, startY_norm, endX_norm, endY_norm], dtype=torch.float32)

        # normalize label in (0, 1, 2) and convert to tensor
        label = torch.tensor(config.LABELS.index(label))

        # apply image transformations if any
        if self.transforms:
            image = self.transforms(image)

        # return a tuple of the images, labels, and bounding box coordinates
        return image, label, bbox

    def __len__(self) -> int:
        # return the size of the dataset
        return len(self.data)
