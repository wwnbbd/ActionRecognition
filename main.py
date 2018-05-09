import torch
import time
from model import Net
from config import pretrained_params_path
import torch.optim as optim
from generateDataBatch import somethingBatch
from config import *
import torch.nn as nn
import time
from utils import load_pretrain_model


#pytorch在LSTM上有很多坑
#variable 最好在输入网络的时候进行转换，否则就会进入计算图
#data_in = torch.randn((45,3,224,224), requires_grad=False)#一定要写volatile=True且forward的时候一定要,还要加eval（）这样训练的时候一定加train,

#define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#get model
model_something = Net(dataset_name="Something",dropout_ratio=0.7)
load_pretrain_model(model_something)
#get data generator
something_something = somethingBatch(datasets_path["SomethingLabel"],datasets_path["SomethingTrain"],datasets_path["SomethingTest"],datasets_path["SomethingValidation"],datasets_path["SomethingData"])

#use multigpu or not
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_something = nn.DataParallel(model_something)

#transfer to GPUs
model_something.to(device)

#define optimzer
optimizer = optim.SGD(model_something.parameters(), lr=0.01)

#define loss function
criterion = nn.CrossEntropyLoss()

#training
for i in range(10000):
    start = time.time()
    optimizer.zero_grad()
    input_data,target_data = something_something.get_training_batch(10)
    input_data = input_data.to(device)
    target_data = target_data.to(device)
    low_out, high_out, space_out = model_something(input_data)
    loss_low = criterion(low_out,target_data)
    loss_high = criterion(high_out,target_data)
    loss_space = criterion(space_out,target_data)
    loss = loss_low + loss_high + loss_space
    loss.backward()
    print(loss.item())
    optimizer.step()
    end = time.time()
    print("Iteration {} : costs {} seconds".format(i, end-start))




