from config import *

class somethingBatch():
    def __init__(self, label, train, test, validation):
        self.labels = self._parse_labels(label)#type: dict
        self.training_sample = self._parse_train(train)#type: dict
        self.validation_sample = self._parse_train(validation)#type: dict
        self.test_sample = self._parse_test(test)#type: list


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


test = somethingBatch(datasets_path["SomethingLabel"],datasets_path["SomethingTrain"],datasets_path["SomethingTest"],datasets_path["SomethingValidation"])

print(test.training_sample["100218"])
print(test.training_sample["48032"])
print(test.validation_sample["85"])
print(test.validation_sample["1753"])
print(test.validation_sample["90751"])
print(test.validation_sample["24413"])