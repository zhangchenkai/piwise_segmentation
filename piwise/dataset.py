import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class VOCTrain(Dataset):

    def __init__(self, root_dir, input_transform=None, target_transform=None):
        set_root = os.path.join(root_dir, 'ImageSets', 'All')

        with open(set_root + '/train_supervised.txt', 'r') as f:
            self.filenames = f.read().splitlines()
        self.filenames.sort()

        self.images_root = os.path.join(root_dir, 'Images')
        self.labels_root = os.path.join(root_dir, 'SegmentationClass')

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class VOCTest(Dataset):

    def __init__(self, root_dir, class_id, input_transform=None, target_transform=None):
        set_root = os.path.join(root_dir, 'ImageSets', 'Patterns')

        with open(set_root + '/p%d_test.txt' % class_id, 'r') as f:
            self.filenames = f.read().splitlines()
        self.filenames.sort()

        self.images_root = os.path.join(root_dir, 'Images')
        self.labels_root = os.path.join(root_dir, 'SegmentationClass')

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        try:
            with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
                label = load_image(f).convert('P')
        except FileNotFoundError:
            mask = np.zeros([image.size[1], image.size[0]], dtype=np.uint8)
            label = Image.fromarray(mask)

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)
