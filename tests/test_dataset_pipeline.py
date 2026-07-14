import os
import tempfile
import shutil
import numpy as np
from PIL import Image

from dataset import get_cifar10_datasets, ImageOnlyDataset


def create_fake_cifar_folder(base_dir, classes=("cat", "dog"), train_per_class=3, test_per_class=2, img_size=32):
    """
    Builds a tiny ImageFolder-structured fake dataset:
        base_dir/train/<class>/*.png
        base_dir/test/<class>/*.png
    Each image is random noise, just enough to exercise the pipeline.
    """
    for split, count in [("train", train_per_class), ("test", test_per_class)]:
        for cls in classes:
            class_dir = os.path.join(base_dir, split, cls)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(count):
                arr = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(os.path.join(class_dir, f"{cls}_{i}.png"))


def test_dataset_pipeline():
    tmp_dir = tempfile.mkdtemp()
    try:
        create_fake_cifar_folder(tmp_dir, classes=("cat", "dog"), train_per_class=3, test_per_class=2)

        train_set, val_set, test_set = get_cifar10_datasets(root=tmp_dir, train_val_split=0.8)

        # 2 classes x 3 images = 6 total train images, split 0.8/0.2 -> 4 train, 2 val (rounded)
        total_train = 6
        expected_n_train = int(0.8 * total_train)
        expected_n_val = total_train - expected_n_train

        assert len(train_set) == expected_n_train, \
            f"Expected {expected_n_train} train samples, got {len(train_set)}"
        assert len(val_set) == expected_n_val, \
            f"Expected {expected_n_val} val samples, got {len(val_set)}"
        print(f"[OK] train/val split sizes correct: train={len(train_set)}, val={len(val_set)}")

        # 2 classes x 2 images = 4 total test images
        assert len(test_set) == 4, f"Expected 4 test samples, got {len(test_set)}"
        print(f"[OK] test set size correct: test={len(test_set)}")

        # check no overlap between train and val indices (random_split guarantees this,
        # but worth confirming the Subset objects are actually disjoint)
        train_indices = set(train_set.indices)
        val_indices = set(val_set.indices)
        assert train_indices.isdisjoint(val_indices), "train_set and val_set overlap!"
        print("[OK] train/val split has no overlapping samples")

        # check raw (train_set) image + label shape/range
        img, label = train_set[0]
        assert img.shape == (3, 32, 32), f"Expected (3,32,32), got {img.shape}"
        assert img.min() >= -1.01 and img.max() <= 1.01, \
            f"Expected range [-1,1], got [{img.min():.3f}, {img.max():.3f}]"
        assert isinstance(label, int), f"Expected int label, got {type(label)}"
        print(f"[OK] raw dataset sample correct: shape={tuple(img.shape)}, "
              f"range=[{img.min():.3f}, {img.max():.3f}], label={label}")

        # now test ImageOnlyDataset wrapping
        wrapped_train = ImageOnlyDataset(train_set)
        assert len(wrapped_train) == len(train_set), "ImageOnlyDataset length mismatch"

        sample = wrapped_train[0]
        assert isinstance(sample, tuple) and len(sample) == 1, \
            f"Expected a 1-tuple (img,), got {sample}"
        assert sample[0].shape == (3, 32, 32), f"Expected (3,32,32), got {sample[0].shape}"
        print(f"[OK] ImageOnlyDataset correctly strips label, yields (img,) of shape {tuple(sample[0].shape)}")

    finally:
        shutil.rmtree(tmp_dir)
        print("[OK] temp fake dataset cleaned up")


if __name__ == "__main__":
    print("===== DATASET PIPELINE TEST (fake ImageFolder data) =====")
    test_dataset_pipeline()
    print("===== ALL TESTS PASSED =====")