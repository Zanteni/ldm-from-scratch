import torch
from dataset import get_cifar10_datasets, ImageOnlyDataset
from utils.transformer import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one


def test_normalize_roundtrip():
    x = torch.tensor([0.0, 0.5, 1.0])
    normed = normalize_to_neg_one_to_one(x)
    assert torch.allclose(normed, torch.tensor([-1.0, 0.0, 1.0])), f"Got {normed}"
    print("[OK] normalize_to_neg_one_to_one correct:", normed.tolist())

    restored = unnormalize_to_zero_to_one(normed)
    assert torch.allclose(restored, x), f"Got {restored}"
    print("[OK] unnormalize_to_zero_to_one correctly inverts:", restored.tolist())


def test_imports_callable():
    assert callable(get_cifar10_datasets), "get_cifar10_datasets is not callable"
    assert callable(ImageOnlyDataset), "ImageOnlyDataset is not callable"
    print("[OK] get_cifar10_datasets and ImageOnlyDataset imported and callable")


def run_dataset_import_suite():
    print("===== DATASET IMPORT/SANITY TEST =====")
    test_normalize_roundtrip()
    test_imports_callable()
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_dataset_import_suite()