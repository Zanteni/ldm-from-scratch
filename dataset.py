import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from utils.transformer import normalize_to_neg_one_to_one


def get_cifar10_datasets(root, train_val_split=0.9):
    """
    root: parent folder containing 'train/' and 'test/' subfolders,
    each organized as <class_name>/*.png (ImageFolder format).

    Returns (train_set, val_set, test_set), each yielding (image, label),
    image in [-1, 1], shape (3, 32, 32).
    """
    # TODO: build the transform pipeline:
    #   transforms.ToTensor()  -> [0,1]
    #   normalize_to_neg_one_to_one  -> [-1,1]
    #   (wrap both in transforms.Compose([...]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_to_neg_one_to_one
    ])

    # TODO: full_train = datasets.ImageFolder(root=f"{root}/train", transform=...)
    full_train = datasets.ImageFolder(root=f"{root}/train",transform=transform)
    # TODO: test_set   = datasets.ImageFolder(root=f"{root}/test", transform=...)
    test_set = datasets.ImageFolder(f"{root}/test",transform=transform)
    # TODO: compute n_train, n_val from len(full_train) and train_val_split
    n_train = int(train_val_split*len(full_train))
    n_val = len(full_train)-n_train

    # TODO: train_set, val_set = torch.utils.data.random_split(full_train, [n_train, n_val])
    train_set,val_set = torch.utils.data.random_split(full_train,[n_train,n_val])
    # TODO: return train_set, val_set, test_set
    return train_set,val_set,test_set


class ImageOnlyDataset(Dataset):
    """Wraps a labeled dataset to yield only the image tensor, discarding
    the label -- VAE training (Phase 1) is unconditional."""

    def __init__(self, base_dataset):
        # TODO: store base_dataset
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        # TODO: get (img, label) from base_dataset[idx], return (img,) only
        img,label = self.base_dataset[idx]
        return (img,)