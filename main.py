import torch
import time
from model import Net
from config import pretrained_params_path
import torch.optim as optim
from generateDataBatch import somethingBatch
from config import *
import torch.nn as nn
import time
from utils import load_pretrain_model,save_checkpoint, load_checkpoint
import argparse
import warnings
warnings.filterwarnings("ignore")


#define parser
parser = argparse.ArgumentParser(description='training hyper parameters')
parser.add_argument("--batch-size", default=8, type=int, help="the number of videos")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate for adam")
parser.add_argument("--loss-ratio-temporal", default=1, type=int, help="ratio between temporal and space")
parser.add_argument("--loss-ratio-space", default=1, type=int, help="artio for space loss")
parser.add_argument("--check-iter", default=100, type=int, help="iteration before save to checkpoint")
parser.add_argument("--pretrain", default="resnet50", type=str, help="from pretrained or finetune")
parser.add_argument("--multigpu", default=False, type=bool, help="wheather use multigpu")
parser.add_argument("--epoch", default=3, type=int, help="number of training epoches")

args = parser.parse_args()

#variable 最好在输入网络的时候进行转换，否则就会进入计算图
#data_in = torch.randn((45,3,224,224), requires_grad=False)#一定要写volatile=True且forward的时候一定要,还要加eval（）这样训练的时候一定加train,

#define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#get model
model_something = Net(dataset_name="Something",dropout_ratio=0.7)
if args.pretrain == "resnet50":
    load_pretrain_model(model_something)
else:
    load_checkpoint(model_something, check_points_path+args.pretrain)

#get data generator
something_loader = torch.utils.data.DataLoader(
    somethingBatch(datasets_path["SomethingLabel"],datasets_path["SomethingTrain"], datasets_path["SomethingData"]),
    batch_size=args.batch_size, shuffle=True, pin_memory=True, number_workers=12)

#use multigpu or not
if (torch.cuda.device_count() > 1) and (args.multigpu == True):
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_something = nn.DataParallel(model_something)

#transfer to GPUs
model_something.to(device)

#define optimzer
optimizer = optim.Adam(model_something.parameters(), lr=args.lr,weight_decay=0.00001)

#define loss function
criterion = nn.CrossEntropyLoss()

#training
for i in range(args.epoch):
    for batch_number, batch_data in enumerate(something_loader):
        start = time.time()
        optimizer.zero_grad()

        input_data,target_data = batch_data["images"],batch_data["labels"]
        input_data = input_data.view(-1,3,224,224).contiguous().type(torch.float32)
        target_data = target_data.view(-1).contiguous()

        input_data = input_data.to(device)
        target_data = target_data.to(device)

        low_out, high_out, space_out = model_something(input_data)
        loss_low = criterion(low_out,target_data)
        loss_high = criterion(high_out,target_data)
        loss_space = criterion(space_out,target_data)
        loss = args.loss_ratio_temporal*loss_low + args.loss_ratio_temporal*loss_high + args.loss_ratio_space*loss_space
        loss.backward()
        print(loss.item())
        optimizer.step()

        end = time.time()
        print("epoch:{}---iter:{}---time:{}".format(i,batch_number,end-start))



        #save parameters
        if i % args.check_iter == 0:
            parameter_path = check_points_path + str(i) +".pth"
            save_checkpoint(model_something, parameter_path)

        #change learning rate




