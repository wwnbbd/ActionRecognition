from config import *
import os
import random
from PIL import Image,ImageOps
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import time
import warnings
import numbers
warnings.filterwarnings("ignore")

#basic transform class
class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images

class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group



class somethingBatch(Dataset):
    def __init__(self, label, train, datapath):
        self.labels = self._parse_labels(label)#type: dict
        self.training_sample = self._parse_train(train)#type: dict
        self.training_sample_number = len(self.training_sample)
        self.training_list = list(self.training_sample.keys())

    def _parse_labels(self, label):
        eng2num = dict()#generate dict
        num2eng = dict()
        with open(label) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i] != "":
                    eng2num[lines[i]] = i
                    num2eng[i] = lines[i]
        return [eng2num, num2eng]#NOTES: number starting from 0 but in file, starting frome 1

    def _parse_train(self, train):#can be used to parse validation file
        video_dir_id_pair = dict()
        with open(train) as f:
            lines = f.readlines()
            for line in lines:
                if line != "":
                    parts = line.split(";")
                    video_dir_id_pair[parts[0]] = self.labels[0][parts[1]] 
        return video_dir_id_pair

    def __len__(self):
        return self.training_sample_number


    def __getitem__(self, index):
        #first count the number of frames in the folder
        video_path = datasets_path["SomethingData"] + self.training_list[index] + "/"
        all_files = os.listdir(video_path)
        selected_files = random.sample(all_files, number_of_frames_per_video)
        selected_files.sort()#all the selected files are in ascending order

        #read in all the sampled frames
        minivideo = []
        for i in range(len(selected_files)):
            minivideo.append(Image.open(video_path + selected_files[i]))

        rescale = GroupScale(256)
        crop = GroupRandomCrop(224)
        flip = GroupRandomHorizontalFlip()

        minivideo = rescale(minivideo)
        minivideo = crop(minivideo)
        minivideo = filp(minivideo)
        
        tsfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406]*15, std=[0.229, 0.224, 0.225]*15)])
        #torch tensor default require grad is false
        minivideo = map(tsfm, minivideo)#normalization mean and std which is from imagenet
        #add extra axis torch stack will add new axis to dim 0
        minivideo = torch.stack(minivideo).contiguous()

        label = torch.LongTensor([self.training_sample[self.training_list[index]]])
        #the shape of minivideo is 15 * 3 *224 *224
        return {"images":minivideo, "labels":label}

'''
test = somethingBatch(datasets_path["SomethingLabel"],datasets_path["SomethingTrain"],datasets_path["SomethingData"])
print(test.training_sample["100218"])
print(test.training_sample["48032"])
'''




        





