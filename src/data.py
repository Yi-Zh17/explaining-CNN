"""Script holds data loader methods.
"""
import argparse

import torch
import torchvision





def get_data_loader() -> torch.utils.data.DataLoader:
    """Creates dataloader for networks from PyTorch's Model Zoo.

    Data loader uses mean and standard deviation for ImageNet.

    Args:
        config: Argparse namespace object.

    Returns:
        Data loader object.

    """
    # Custom ImageFolder
    class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
        def __getitem__(self, index):
            # get original tuple (image, label)
            original_tuple = super().__getitem__(index)
            # get image path
            path = self.imgs[index][0]
            # make new tuple (image, label, path)
            tuple_with_path = original_tuple + (path,)
            return tuple_with_path

    
    input_dir = "./input/"
    batch_size = 1

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms = []


    transforms += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]

    transform = torchvision.transforms.Compose(transforms=transforms)
    #dataset = torchvision.datasets.ImageFolder(root=input_dir, transform=transform)
    dataset = ImageFolderWithPaths(root=input_dir, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return data_loader
