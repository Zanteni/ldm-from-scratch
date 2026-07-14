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
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_to_neg_one_to_one,
    ])

    full_train = datasets.ImageFolder(root=f"{root}/train", transform=transform)
    test_set = datasets.ImageFolder(root=f"{root}/test", transform=transform)

    n_train = int(train_val_split * len(full_train))
    n_val = len(full_train) - n_train
    train_set, val_set = torch.utils.data.random_split(full_train, [n_train, n_val])

    return train_set, val_set, test_set


class ImageOnlyDataset(Dataset):
    """Wraps a labeled dataset to yield only the image tensor, discarding
    the label -- VAE training (Phase 1) is unconditional."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return (img,)