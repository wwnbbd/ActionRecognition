import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
from generateDataBatch import somethingBatch
from model import Net
from utils import load_checkpoint
from config import *
import argparse
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="parse argument for evaluation")
parser.add_argument("--batch-size", default=4, type=int, help="batch size")
parser.add_argument("--multigpu", default=False, type=bool, help="using multigpu or not")
parser.add_argument("--check1", default="None", type=str, help="which checkpoint to use")
parser.add_argument("--check2", default="None", type=str, help="the second checkpoint")
args = parser.parse_args()

counter = []
correct = []
confusion = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


testSomething = Net(dataset_name="Something")
if args.check1== "None" or args.check2 == "None":
    raise Exception("Must give checkpoint path")
for para in range(2):
    if para == 0:
        load_checkpoint(testSomething,check_points_path+args.check1)
    else:
        load_checkpoint(testSomething,check_points_path+args.check2)

    if (torch.cuda.device_count() > 1) and args.multigpu:
        print("using mutigpu")
        testSomething = torch.nn.DataParallel(testSomething)


    testSomething.to(device)
    testSomething.eval()

    something_validation_loader = torch.utils.data.DataLoader(
        somethingBatch(datasets_path["SomethingLabel"],datasets_path["SomethingValidation"], datasets_path["SomethingData"]),
        batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)


    top1 = 0
    top5 = 0
    total = 0
    top_arg = 0
    class_counter_list = [0]*174
    class_correct_list = [0]*174
    y_true = []
    y_predict = []
    with torch.set_grad_enabled(False):
        for batch_number,batch_data in enumerate(something_validation_loader):
            print("----{}-------".format(batch_number))
            input_data, target_data = batch_data["images"], batch_data["labels"]
            target_data = target_data.view(-1)
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            y_true.append(target_data.cpu().numpy())
            total += target_data.size()[0]
            for i in range(target_data.size()[0]):
                class_counter_list[target_data[i].item()] += 1
            low_out, space_out = testSomething(input_data)
            #low_out = (low_out + space_out)/2
            if para == 1: space_out = low_out
            index = torch.sort(space_out, 1, descending=True)[1]
            argmax = torch.argmax(low_out,1)
            for i in range(5):
                if i == 0:
                    y_predict.append(index[:,i].cpu().numpy())
                    top1 += torch.sum(index[:,i] == target_data).item()
                    ifequal = index[:,i] == target_data
                    for j in range(index.size()[0]):
                        if ifequal[j].item() == 1:
                            class_correct_list[target_data[j].item()] += 1
                    top5 += torch.sum(index[:,i] == target_data).item()
                    top_arg += torch.sum(argmax == target_data).item()
                else:
                    top5 += torch.sum(index[:,i] == target_data).item()
    print(top1)
    print(top5)
    print(total)
    print(top_arg)
    correct.append(class_correct_list)
    counter.append(class_counter_list)
    y_true = np.hstack(y_true)
    y_predict = np.hstack(y_predict)
    confusion.append(confusion_matrix(y_true,y_predict))


#print compare results
#class_acc = []
for i in range(174):
    #class_acc.append(class_correct_list[i]/class_counter_list[i])
    print("class id: {} \t  acc1: {:0.3f} \t \t acc2: {:0.3f}".format(i,correct[0][i]/counter[0][i], correct[1][i]/counter[1][i]))

print(np.argsort(confusion[0],1)[:,-5:-1])
print(np.argsort(confusion[1],1)[:,-5:-1])
