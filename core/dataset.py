import os
import cv2
import io
import glob
import scipy
import json
import zipfile
import random
import collections
import torch
import math
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from skimage.color import rgb2gray, gray2rgb
from core.utils import ZipReader, create_fixed_rectangular_mask, create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip, Normalize


class MUSICDataset(Dataset):
    def __init__(self, dataset_args: dict, split='train'):
        self.dataset_args = dataset_args
        self.root_dir = dataset_args["root_dir"]
        self.ref_frames = dataset_args["sample_length"]
        self.image_width, self.image_height = dataset_args["w"], dataset_args["h"]
        self.image_shape = (self.image_width, self.image_height)

        self.video_dict = dict()
        for video_id in os.listdir(f"{self.root_dir}/png"):
            self.video_dict[video_id] = len(os.listdir(f"{self.root_dir}/png/{video_id}"))
        self.video_ids = sorted(list(self.video_dict.keys()))

        self.hflipper = transforms.RandomHorizontalFlip(1.)
        self.image_transforms = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
            Normalize()
        ])
        self.mask_transforms = transforms.Compose([
            Stack(),
            ToTorchFormatTensor()
        ])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):  # (B, T, C, H, W)
        video_id = self.video_ids[index]
        all_frames = [f"{str(i).zfill(5)}.png" for i in range(1, self.video_dict[video_id] + 1)]
        sampled_idxs = self.get_frame_index(len(all_frames), self.ref_frames)
        frames = list()
        masks = create_fixed_rectangular_mask(5, 256, 256, 42)
        
        for i in sampled_idxs:
            image_path = f"{self.root_dir}/png/{video_id}/{all_frames[i]}"
            image = Image.open(image_path)
            assert image.size == self.image_shape
            frames.append(image)
        if random.uniform(0, 1) > 0.5:
            frames = [self.hflipper(frame) for frame in frames]

        frame_tensors = self.image_transforms(frames)  # (T, C, H, W)
        mask_tensors = self.mask_transforms(masks)
        data_dict = frame_tensors, mask_tensors
        return data_dict

    # Index sampling function
    def get_frame_index(self, length, ref_count):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), ref_count)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-ref_count)
            ref_index = [pivot+i for i in range(ref_count)]
        return ref_index


class AVEDataset(Dataset):
    def __init__(self, args: dict, split="train"):
        self.args = args
        self.data_root = args["data_root"]
        self.mask_dir = args["mask_root"]
        self.split = split
        self.ref_count = args["sample_length"]
        self.image_shape = self.image_width, self.image_height = (args["w"], args["h"])
        assert self.split in ["train", "val", "test"]

        self.video_dict = dict()
        for video_id in os.listdir(f"{self.data_root}/{split}/image"):
            self.video_dict[video_id] = len(os.listdir(f"{self.data_root}/{split}/image/{video_id}"))
        self.video_ids = list(self.video_dict.keys())

        self.vflipper = transforms.RandomVerticalFlip(1.)
        self.hflipper = transforms.RandomHorizontalFlip(1.)

        self.mask_transforms = transforms.Compose([
            Stack(),
            ToTorchFormatTensor()
        ])
        self.image_transforms = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
            Normalize(0.5, 0.5)
        ])
    
    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        all_frames = [f"{str(i).zfill(3)}.png" for i in range(self.video_dict[video_id])]
        all_masks = create_random_shape_with_random_motion(len(all_frames), self.image_height, self.image_width)
        mask_path = f"{self.mask_dir}/{str(random.randrange(0, 12000)).zfill(5)}.png"
        mask = Image.open(mask_path).resize((self.image_height, self.image_width)).convert("L")
        if random.uniform(0, 1) > 0.5:
            mask = self.hflipper(mask)
        if random.uniform(0, 1) > 0.5:
            mask = self.vflipper(mask)
        all_masks = [mask] * len(all_frames)
        ref_index = self.get_ref_index(len(all_frames), self.ref_count)

        frames = list()
        masks = list()
        for i in ref_index:
            image_path = f"{self.data_root}/{self.split}/image/{video_id}/{all_frames[i]}"
            image = Image.open(image_path)
            if image.size != self.image_shape:
                image = image.resize(self.image_shape)
            frames.append(image)
            masks.append(all_masks[i])
        
        if self.split == "train":
            if random.uniform(0, 1) > 0.5:
                hflip = transforms.RandomHorizontalFlip(1.)
                frames = [hflip(frame) for frame in frames]
        
        frame_tensors = self.image_transforms(frames)
        mask_tensors = self.mask_transforms(masks)
        return frame_tensors, mask_tensors

    def get_ref_index(self, length, ref_count):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), ref_count)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-ref_count)
            ref_index = [pivot+i for i in range(ref_count)]
        return ref_index


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])

        with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug or split != 'train':
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(self.video_dict[video_name])]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        masks = []
        for idx in ref_index:
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name), all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            frames = GroupRandomHorizontalFlip()(frames)
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
