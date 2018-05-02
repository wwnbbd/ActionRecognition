import torch
import time
from model import Net
from config import pretrained_params_path

#pytorch在LSTM上有很多坑

def load_pretrain_model(model,basenet="resnet50"):#这个函数暂时认为是正确的，不同的测试方式得到了完全相反的实验结果（"basenet.conv1.weight"）
    params = torch.load(pretrained_params_path[basenet]) #ordered dict
    #tmp1 = params["conv1.weight"]#------------
    model_dict = model.state_dict()
    #tmp2 = model_dict["basenet.conv1.weight"]#-------------
    pretrained_dict = {"basenet."+k:v for k,v in params.items() if "basenet."+k in model_dict}
    model_dict.update(pretrained_dict)#一部分value的类型是parameters一部分是float tensor
    model.load_state_dict(model_dict)
    #tmp3 = model.state_dict()["basenet.conv1.weight"]#----------
    #print(tmp1.data.size())#-----------
    #print(tmp1.data[2,2,0:3,0:3])#------------
    #print("===========================")#-----------
    #print(tmp2[2,2,0:3,0:3])#-------------
    #print("===========================")#------------
    #print(tmp3[2,2,0:3,0:3])#-------------


def load_checkpoint(model,params_path):#can be used to load checkpoints
    model.load_state_dict(torch.load(params_path))

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)



#variable 最好在输入网络的时候进行转换，否则就会进入计算图
data_in = torch.randn((45,3,224,224), requires_grad=False)#一定要写volatile=True且forward的时候一定要,还要加eval（）这样训练的时候一定加train,

c = Net(dataset_name="UCF",dropout_ratio=0.7)

with torch.set_grad_enabled(False):
    c.eval()
    a1,b1,c1=c(data_in)
    load_pretrain_model(c)
    _,_,c2 = c(data_in)
#save_checkpoint(c,'./01.pth')
print(a1.size())
print(b1.size())
print(c1.size())
print(c2-c1)


