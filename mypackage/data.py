from jdit.dataset import DataLoadersFactory
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image, ImageChops
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import os
import random
import platform
import torch
import numpy as np
import math

# import cv2
sysstr = platform.system()
if (sysstr == "Windows"):
    ROOT_PATH = r"E:\dataset\segmentationtraininglib"
elif (sysstr == "Linux"):
    ROOT_PATH = "/data/dgl/segmentationtraininglib"
else:
    raise TypeError("can't not support platform `%s`" % sysstr)

# input
IMAGE_PATH = os.path.join(ROOT_PATH, "image")

MASK_PATH_DIC = {"gaussian": os.path.join(ROOT_PATH, "gaussianMask"),
                 "circular": os.path.join(ROOT_PATH, "circularMask"),
                 "smallCircular": os.path.join(ROOT_PATH, "smallcircularMask")
                 }

NOISE_PATH_DIC = {"nN": os.path.join(ROOT_PATH, "noNoise"),
                  "nN_nBG": os.path.join(ROOT_PATH, "noBackgroundnoNoise"),
                  "nN_nBG_SR": os.path.join(ROOT_PATH, "noNoiseNoBackgroundSuperresolution"),
                  "nN_nBG_UP2X": os.path.join(ROOT_PATH, "noNoiseNoBackgroundUpinterpolation2x"),
                  "nN_UP2X": os.path.join(ROOT_PATH, "noNoiseUpinterpolation2x")}
# anti
IMAGE_TEST_PATH = os.path.join(ROOT_PATH, "Standard test images_SNR")

assert os.path.exists(IMAGE_PATH), "can not find %s" % IMAGE_PATH
assert os.path.exists(IMAGE_TEST_PATH), "can not find %s" % IMAGE_TEST_PATH
# print(IMAGE_PATH, os.path.exists(IMAGE_PATH))
for i in MASK_PATH_DIC:
    assert os.path.exists(MASK_PATH_DIC[i]), "can not find %s" % IMAGE_PATH

image_dir_path = IMAGE_PATH
mask_dir_path = MASK_PATH_DIC["gaussian"]
test_dir_path = IMAGE_TEST_PATH


class AtomDatasets(DataLoadersFactory):
    def __init__(self, image_dir_path, mask_dir_path, test_dir_path, batch_size, valid_size, num_workers=-1,
                 shuffle=True,
                 subdata_size=0.01,
                 use_LM=True):
        self.valid_size = valid_size
        self.image_dir_path = image_dir_path
        self.mask_dir_path = mask_dir_path
        self.test_dir_path = test_dir_path
        self.use_LM = use_LM
        super(AtomDatasets, self).__init__("", batch_size, num_workers, shuffle, subdata_size)

    def build_transforms(self, resize: int = 256):
        # self.train_transform_list_input = [
        #     transforms.RandomRotation(180, resample=Image.NEAREST  )  ,
        #     transforms.RandomResizedCrop(
        #             256, scale=(0.25, 1.0), ratio=(1, 1), interpolation=2),
        #     transforms.RandomApply([transforms.Compose([
        #         transforms.Pad(128),
        #         transforms.RandomResizedCrop(256, scale=(1.0, 1.0), ratio=(1, 1), interpolation=2)])]
        #             , p=0.9),
        #     transforms.Resize(256),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5])
        #     # _Cutout(2, 50)
        #     ]
        # self.train_transform_list_real = [
        #     transforms.RandomRotation(180, resample=Image.NEAREST),
        #     transforms.RandomResizedCrop(
        #             256, scale=(0.25, 1.0), ratio=(1, 1), interpolation=2),
        #     transforms.RandomApply([
        #         transforms.Pad(128),
        #         transforms.RandomResizedCrop(256, scale=(0.25, 1.0), ratio=(1, 1), interpolation=2)]),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5])
        #     ]
        self.test_transform_list = self.vaild_transform_list = [
            transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
            transforms.Normalize([0.5], [0.5])
            ]

    def build_datasets(self):
        imagesNames = None
        maskNames = None
        for root, dirs, files in os.walk(self.image_dir_path):
            imagesNames = files
            break
        for root, dirs, files in os.walk(self.mask_dir_path):
            maskNames = files
            break
        assert (len(imagesNames) == len(maskNames))
        total = len(imagesNames)

        dataset_total = _TrainDataset(imagesNames, maskNames,
                                      self.image_dir_path,
                                      self.mask_dir_path, self.use_LM)
        if isinstance(self.valid_size, float):
            assert self.valid_size > 0 and self.valid_size < 1, "`valid_size` must between 0 and 1"
            valid_size = int(self.valid_size * total)
        elif isinstance(self.valid_size, int):
            valid_size = self.valid_size
            assert valid_size < total, "`valid_size` must smaller than total (%d)" % total
        else:
            raise TypeError("`valid_size` must be float or int")

        self.dataset_train, self.dataset_valid = random_split(dataset_total, [total - valid_size, valid_size])
        self.dataset_train = DiffSubset(self.dataset_train, True)
        self.dataset_valid = DiffSubset(self.dataset_valid, False)

        print("train size:%d    valid size:%d" % (total - valid_size, valid_size))
        self.dataset_test = _TestDataset(self.test_dir_path, self.test_transform_list, min_size=32)

    def build_loaders(self):
        super(AtomDatasets, self).build_loaders()
        self.loader_test = DataLoader(self.dataset_test, batch_size=1, shuffle=False)
        self.nsteps_test = len(self.loader_test)


class DiffSubset(object):
    def __init__(self, subset, train):
        self.subset = subset
        self.train = train

    def __getattr__(self, item):
        # self.subset.dataset.train = self.train
        return getattr(self.subset, item)

    def __getitem__(self, idx):
        self.subset.dataset.train = self.train
        return self.subset.dataset[self.subset.indices[idx]]

    def __len__(self):
        return len(self.subset.indices)


class _TrainDataset(Dataset):
    def __init__(self, x_file_names, y_file_names, x_dir_path, y_dir_path, use_LM=True):
        self.x = []
        self.y = []
        self.x_dir_path = x_dir_path
        self.y_dir_path = y_dir_path
        self.x_file_names = x_file_names
        self.y_file_names = y_file_names
        self.nums = len(x_file_names)
        self.train = True
        self.use_LM = use_LM

    def __len__(self):
        return self.nums

    def transform(self, image, mask):
        # 测试集不做数据增强
        # angle = transforms.RandomRotation.get_params([-180, 180])
        # image = tf.rotate(image, angle, resample=Image.NEAREST)
        # mask = tf.rotate(mask, angle, resample=Image.NEAREST)
        if self.train:
            angle = transforms.RandomRotation.get_params([-180, 180])
            image = tf.rotate(image, angle, resample=Image.NEAREST)
            mask = tf.rotate(mask, angle, resample=Image.NEAREST)

            if self.use_LM:
                if random.random() > 0.5:
                    image = RandomLinearLightMask(random_reverse=True)(image)
                if random.random() > 0.5:
                    image = RandomPointLightMask(random_reverse=True)(image)

            if random.random() > 0.5:
                image = tf.hflip(image)
                mask = tf.hflip(mask)
            if random.random() > 0.5:
                image = tf.vflip(image)
                mask = tf.vflip(mask)
            if random.random() > 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                        image, scale=(0.25, 1.0), ratio=(1, 1))
                image = tf.resized_crop(image, i, j, h, w, 256)
                mask = tf.resized_crop(mask, i, j, h, w, 256)
            else:
                pad = random.randint(0, 192)
                image = tf.pad(image, pad)
                image = tf.resize(image, 256)
                mask = tf.pad(mask, pad)
                mask = tf.resize(mask, 256)

        image = tf.to_tensor(image)
        image = tf.normalize(image, [0.5], [0.5])
        mask = tf.to_tensor(mask)
        mask = tf.normalize(mask, [0.5], [0.5])
        return image, mask

    def __getitem__(self, index):
        X_IMG_URL = os.path.join(self.x_dir_path, self.x_file_names[index])
        Y_IMG_URL = os.path.join(self.y_dir_path, self.y_file_names[index])
        with Image.open(X_IMG_URL) as img:
            x_img = img.convert("L")
        with Image.open(Y_IMG_URL) as img:
            y_img = img.convert("L")

        x, y = self.transform(x_img, y_img)

        return x, y


class _TestDataset(Dataset):
    def __init__(self, test_dir_path, transform_list=None, min_size=32):
        self.test_img = []
        self.imagesNames = []
        self.min_size = min_size
        self.test_dir_path = test_dir_path
        if transform_list is None:
            transform_list = [
                transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
                transforms.Normalize([0.5], [0.5])
                ]
        self.transform = transforms.Compose(transform_list)

        for root, dirs, files in os.walk(test_dir_path):
            self.imagesNames = files
            break
        self.nums = len(self.imagesNames)

    def __len__(self):
        return self.nums

    def __getitem__(self, index):

        IMG_URL = os.path.join(self.test_dir_path, self.imagesNames[index])
        with Image.open(IMG_URL) as img:
            img = img.convert("L")

        row, col = img.size
        padding_row = (self.min_size * math.ceil(row / self.min_size) - row)
        padding_col = (self.min_size * math.ceil(col / self.min_size) - col)
        padding_left = math.ceil(padding_row / 2)
        padding_right = math.floor(padding_row / 2)
        padding_top = math.ceil(padding_col / 2)
        padding_bottom = math.floor(padding_col / 2)
        #  left, top, right and bottom borders
        x = transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0)(img)
        x = self.transform(x)
        return x


class TestDatasetWithPadding(_TestDataset):
    def __init__(self, test_dir_path, transform_list=None, min_size=32):
        super(TestDatasetWithPadding, self).__init__(test_dir_path, transform_list, min_size)

    def __getitem__(self, index):
        IMG_URL = os.path.join(self.test_dir_path, self.imagesNames[index])
        with Image.open(IMG_URL) as img:
            img = img.convert("L")

        row, col = img.size
        padding_row = (self.min_size * math.ceil(row / self.min_size) - row)
        padding_col = (self.min_size * math.ceil(col / self.min_size) - col)
        padding_left = math.ceil(padding_row / 2)
        padding_right = math.floor(padding_row / 2)
        padding_top = math.ceil(padding_col / 2)
        padding_bottom = math.floor(padding_col / 2)
        #  left, top, right and bottom borders
        x = transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0)(img)
        x = self.transform(x)
        return x, (padding_left, padding_top, row, col)


class _Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length, mask_numb=.0):
        self.n_holes = n_holes
        self.length = length
        self.mask_numb = mask_numb

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = self.mask_numb

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomLinearLightMask(object):
    def __init__(self, sigma=1, random_reverse=False):
        self.sigma = sigma
        self.random_reverse = random_reverse

    def gauss_map(self, h, w, random_reverse):
        EPCLON = 1e-8
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        k = random.uniform(-center_y / (w - center_x + EPCLON), (h - center_y) / (w - center_x + EPCLON))
        A = k
        B = -1
        C = - k * center_x + center_y
        R = random.uniform(np.sqrt((w / 4) ** 2 + (h / 4) ** 2), (np.sqrt((w / 2) ** 2 + (h / 2) ** 2))) * self.sigma

        x1 = np.expand_dims(np.arange(w), 0)
        y1 = np.expand_dims(np.arange(h), 0).transpose()
        Gauss_map = abs(A * np.repeat(x1, h, 0) + B * np.repeat(y1, w, 1) + C) / np.sqrt(
                A ** 2 + B ** 2)
        Gauss_map = np.exp(-Gauss_map / (R + EPCLON))
        if random.random() > 0.5:
            Gauss_map = Gauss_map.transpose()
        if random_reverse and random.random() > 0.5:
            Gauss_map = abs(Gauss_map - 1)
        return Gauss_map

    def __call__(self, img: Image):
        h = img.size[0]
        w = img.size[1]
        gauss_np = self.gauss_map(h, w, self.random_reverse)
        if img.mode == "L":
            gauss_img = Image.fromarray(np.uint8(gauss_np * 255))
        else:
            raise TypeError("Only be used by PIL.Image.mode = 'L'")
        img = ImageChops.multiply(img, gauss_img)
        return img


# class RandomErode(object):
#     def __init__(self, sigma=1, random_reverse=False):
#         self.sigma = sigma
#         self.random_reverse = random_reverse
#     def __call__(self, img: Image):
#         h = img.size[0]
#         w = img.size[1]
#         gauss_np = self.gauss_map(h, w, self.random_reverse)
#         if img.mode == "L":
#             gauss_img = Image.fromarray(np.uint8(gauss_np * 255))
#         else:
#             raise TypeError("Only be used by PIL.Image.mode = 'L'")
#         img = ImageChops.multiply(img, gauss_img)
#         return img

class RandomPointLightMask(object):
    def __init__(self, sigma=1, random_reverse=True):
        self.sigma = sigma
        self.random_reverse = random_reverse

    def gauss_map(self, h, w, random_reverse):
        EPCLON = 1e-8
        center_x = random.randint(0, h)
        center_y = random.randint(0, w)
        R = random.uniform(np.sqrt((w / 4) ** 2 + (h / 4) ** 2), (np.sqrt((w / 2) ** 2 + (h / 2) ** 2))) * self.sigma

        x1 = np.expand_dims(np.arange(w), 0)
        y1 = np.expand_dims(np.arange(h), 0).transpose()
        Gauss_map_x = (center_x - x1) ** 2
        Gauss_map_y = (center_y - y1) ** 2
        Gauss_map = np.sqrt(np.repeat(Gauss_map_x, h, axis=0) + np.repeat(Gauss_map_y, w, axis=1))
        Gauss_map = np.exp(-Gauss_map / (R + EPCLON))

        if random_reverse and random.random() > 0.5:
            Gauss_map = abs(Gauss_map - 1)

        return Gauss_map

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        gauss_np = self.gauss_map(h, w, self.random_reverse)
        if img.mode == "L":
            gauss_img = Image.fromarray(np.uint8(gauss_np * 255))
        else:
            raise TypeError("Only be used by PIL.Image.mode = 'L'")
        img = ImageChops.multiply(img, gauss_img)
        return img


def CheckLoader(loader):
    count = 0
    for index, batch in enumerate(loader):
        input = batch[0]  # [2,1,256,256]
        real = batch[1]  # [2,3,256,256]
        a = transforms.Normalize([-1], [2])(input[0])
        a = transforms.ToPILImage()(a.reshape(1, 256, 256)).convert("L")
        b = transforms.Normalize([-1], [2])(real[0][0].reshape(1, 256, 256))
        b = transforms.ToPILImage()(b.reshape(1, 256, 256)).convert("L")
        a.show()
        b.show()
        count += 1
        if count == 4:
            break


def checkTestLoader(test_loader):
    count = 0
    for index, batch in enumerate(test_loader):
        real = batch  # [2,3,256,256]
        a = transforms.Normalize([-1], [2])(real[0])
        a = transforms.ToPILImage()(a.reshape(1, 256, 256)).convert("L")
        b = transforms.Normalize([-1], [2])(real[0].reshape(1, 256, 256))
        b = transforms.ToPILImage()(b.reshape(1, 256, 256)).convert("L")
        a.show()
        b.show()
        count += 1
        if count == 4:
            break


if __name__ == '__main__':

    batch_size = 2
    d = AtomDatasets(IMAGE_PATH, MASK_PATH_DIC["gaussian"], IMAGE_TEST_PATH, batch_size, 0.001)
    train_loader = d.loader_train
    valid_loader = d.loader_valid
    test_loader = d.loader_test
    CheckLoader(valid_loader)
    CheckLoader(train_loader)
    CheckLoader(valid_loader)
    # checkTestLoader(test_loader)
