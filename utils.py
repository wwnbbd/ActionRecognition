from config import pretrained_params_path
import torch

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

