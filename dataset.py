import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from utils.transformer import normalize_to_neg_one_to_one
from models.vae import VAE


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
    
"""
       New data set for the latent space
"""
@torch.no_grad()
def build_latent_cache(vae: VAE, dataloader, device="cpu"):
    """
    Runs every image in `dataloader` through the frozen VAE encoder once,
    collecting mu/logvar for the whole dataset.
    Returns (all_mu, all_logvar), each shape (N, z_channels, 8, 8).
    """
    vae.eval()
    vae.to(device=device)

    for param in vae.encoder.parameters():
        param.requires_grad = False

    mu_list, logvar_list = [], []

    for batch in dataloader:
        img_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
        img_batch = img_batch.to(device)

        mu, logvar = vae.encode(img_batch)
        mu_list.append(mu.cpu())
        logvar_list.append(logvar.cpu())

    all_mu = torch.cat(mu_list, dim=0)
    all_logvar = torch.cat(logvar_list, dim=0)

    return all_mu, all_logvar


class LatentDataset(Dataset):
    """
    Wraps precomputed (mu, logvar) tensors. Each __getitem__ call samples a
    FRESH z via reparameterization, so repeated epochs see different latent
    samples for the same underlying image (free augmentation from the VAE's
    own stochasticity).
    """

    def __init__(self, all_mu, all_logvar):
        self.all_mu = all_mu
        self.all_logvar = all_logvar

    def __len__(self):
        return self.all_mu.shape[0]

    def __getitem__(self, idx):
        mu, logvar = self.all_mu[idx], self.all_logvar[idx]
        z = VAE.reparametrize(mu=mu, logvar=logvar)
        return (z,)
    
@torch.no_grad()
def build_conditional_latent_cache(vae, dataloader, device="cpu"):
    """
    Same as build_latent_cache, but also collects the class label for
    each image, needed for class-conditioned diffusion training.
    Returns (all_mu, all_logvar, all_labels).
    """
    # same as build_latent_cache, but each batch is (img_batch, label_batch)
    # instead of (img_batch,) -- collect label_batch into a labels_list too,
    # torch.cat at the end alongside mu/logvar
    vae.eval()
    vae.to(device=device)

    for param in vae.encoder.parameters():
        param.requires_grad = False

    mu_list, logvar_list,labels_list = [], [],[]
    for batch in dataloader:
        img_batch,img_label = batch[0],batch[1]
        img_batch = img_batch.to(device)
        img_label = img_label.to(device)
        mu, logvar = vae.encode(img_batch)
        mu_list.append(mu.cpu())
        logvar_list.append(logvar.cpu())
        labels_list.append(img_label.cpu())
    all_mu = torch.concat(mu_list,dim=0)
    all_logvar = torch.concat(logvar_list,dim=0)
    all_labels = torch.concat(labels_list,dim=0)
    return all_mu,all_logvar,all_labels

    


    

class ConditionalLatentDataset(Dataset):
    """
    Wraps precomputed (mu, logvar, label) tensors. Same fresh-resampling
    behavior as LatentDataset, but __getitem__ also returns the label.
    """
    def __init__(self, all_mu, all_logvar, all_labels):
        self.all_mu = all_mu
        self.all_logvar = all_logvar
        self.all_labels = all_labels
    def __len__(self):
        return self.all_mu.shape[0]
    def __getitem__(self, idx):
        # z = VAE.reparametrize(mu, logvar)
        z = VAE.reparametrize(mu=self.all_mu[idx],logvar=self.all_logvar[idx])
        label = self.all_labels[idx]
        # return (z, label) -- or (z,label) tuple, your call on exact format
        return (z,label)