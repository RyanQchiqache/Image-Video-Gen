import PIL.Image as Image
import glob
from torch.utils.data import Dataset
import os


class CelebADataset(Dataset):
    """
        Lightweight CelebA loader for the aligned image folder (img_align_celeba).

        Unlike torchvision.datasets.CelebA, this class:
          • Loads images directly from a flat directory of .jpg files
          • Does NOT rely on the CelebA metadata CSV files
          • Returns (image, dummy_label) because DCGAN is unsupervised
          • Applies any given transform (crop/resize/normalize)

        This keeps the dataloader simple and fast, and avoids issues where
        torchvision.CelebA expects the dataset to follow the official directory
        structure (list_attr_celeba.txt, identity_CelebA.txt, etc.).
    """
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0