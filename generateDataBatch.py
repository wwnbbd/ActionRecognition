from config import *
import os
import random
from skimage import io,transform
import numpy as np
from torchvision import transforms
import torch
import time 

#basic transform class
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image



class somethingBatch():
    def __init__(self, label, train, test, validation, datapath):
        self.labels = self._parse_labels(label)#type: dict
        self.training_sample = self._parse_train(train)#type: dict
        self.validation_sample = self._parse_train(validation)#type: dict
        self.test_sample = self._parse_test(test)#type: list
        self.last_sample_pos = 0
        self.training_sample_number = len(self.training_sample)
        self.validation_sample_number = len(self.validation_sample)
        self.test_sample_number = len(self.test_sample)
        self.trained_ratio = 0 #trained sample number divide total training sample number, may > 1.0
        self.training_list = list(self.training_sample.keys())
        self.validation_list = list(self.validation_sample.keys())


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

    def _parse_test(self, test):
        result = []
        with open(test) as f:
            lines = f.readlines()
            for line in lines:
                if line != "":
                    result.append(line)
        return result

    #this function return pytorch tensor
    def _sample_single_video(self, videoID):#video Id should be a string
        #first count the number of frames in the folder
        video_path = datasets_path["SomethingData"] + videoID + "/"
        all_files = os.listdir(video_path)
        selected_files = random.sample(all_files, number_of_frames_per_video)
        selected_files.sort()#all the selected files are in ascending order

        #read in all the sampled frames
        start = time.time()
        minivideo = io.imread(video_path + selected_files[0])#shape: height * width * channel
        for i in range(1,len(selected_files)):
            minivideo = np.concatenate((minivideo, io.imread(video_path + selected_files[i])),axis=2)
        end = time.time()
        print("read images using {} seconds".format(end-start))

        rescale = Rescale(256)
        crop = RandomCrop(224)

        minivideo = rescale(minivideo)
        minivideo = crop(minivideo)
        if random.random() > 0.5:
            minivideo = minivideo[:,::-1,:].copy()
        trans1 = time.time()
        print("transforms {} seconds".format(trans1 - end))

        
        #minivideo shape: height * width * channel (also PIL image format)
        tsfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406]*15, std=[0.229, 0.224, 0.225]*15)])
        #torch tensor default require grad is false
        minivideo_transformed = tsfm(minivideo)#normalization mean and std which is from imagenet
        #add extra axis
        minivideo_transformed = minivideo_transformed.unsqueeze_(0).view(-1,3,224,224).contiguous()
        print("trans2 {} seconds".format(time.time() - trans1))
        self.last_sample_pos = (self.last_sample_pos + 1) % self.training_sample_number
        self.trained_ratio += 1/self.training_sample_number
        return minivideo_transformed


    def get_training_batch(self, batch_size):#batch size indicates how many videos in a batch not the number of frames
        data = []
        gt = [0] * batch_size
        for i in range(batch_size):
            gt[i] = self.training_sample[self.training_list[self.last_sample_pos]]
            data.append(self._sample_single_video(self.training_list[self.last_sample_pos]))
        data = torch.cat(data)
        gt = torch.LongTensor(gt)
        data = data.type(torch.float)
        return data, gt


        





