from config import *
import os
import random
from skimage import io
import numpy as np
from torchvision import transforms

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
        minivideo = io.imread(video_path + selected_files[0])#shape: height * width * channel
        for i in range(1,len(selected_files)):
            minivideo = np.concatenate((minivideo, io.imread(selected_files[i])),axis=2)
        
        #minivideo shape: height * width * channel (also PIL image format)
        tsfm = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406]*15, std=[0.229, 0.224, 0.225]*15)])
        #torch tensor default require grad is false
        minivideo_transformed = tsfm(minivideo)#normalization mean and std which is from imagenet
        #add extra axis
        minivideo_transformed.unsqueeze_(0).view(-1,3,224,224).contiguous()

        self.last_sample_pos = (self.last_sample_pos + 1) % self.training_sample_number
        self.trained_ratio += 1/self.training_sample_number
        return minivideo_transformed

    def get_batch(self, batch_size):
        pass

        



test = somethingBatch(datasets_path["SomethingLabel"],datasets_path["SomethingTrain"],datasets_path["SomethingTest"],datasets_path["SomethingValidation"],datasets_path["SomethingData"])

print(test.training_sample["100218"])
print(test.training_sample["48032"])
print(test.validation_sample["85"])
print(test.validation_sample["1753"])
print(test.validation_sample["90751"])
print(test.validation_sample["24413"])

