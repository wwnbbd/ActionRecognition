import math
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet import resnet50
from config import datasets #config


#注意 resnet的实现没有使用bias
class Temporal(nn.Module):#in fact is a bottleneck
    def __init__(self, input_num):
        super(Temporal, self).__init__()
        self.conv1 = nn.Conv2d(input_num, int(input_num/4), kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(input_num/4))
        self.conv2 = nn.Conv2d(int(input_num/4), int(input_num/4), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(input_num/4))
        self.conv3 = nn.Conv2d(int(input_num/4), input_num, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(input_num)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out  = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, frames_per_video = 15, dataset_name = "UCF"):
        super(Net, self).__init__()
        self.frames_per_video = frames_per_video
        self.dataset_name = dataset_name
        self.output_channel = datasets[dataset_name]
        #self.dropout_ratio = dropout_ratio

        self.relu = nn.ReLU()
        self.basenet = resnet50(pretrained=False)

        #used after substraction
        self.sub_conv_low = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
    
        #Temple branch one low level feature
        self.low_layer_node_1 = Temporal(512)
        self.low_subsample_1 = nn.Conv2d(512, 512*2, kernel_size=1, stride=2, bias=False)
        self.low_subsample_1_bn = nn.BatchNorm2d(512*2)

        self.low_layer_node_2 = Temporal(512*2)
        self.low_subsample_2 = nn.Conv2d(512*2, 512*4, kernel_size=1, stride=2, bias=False)
        self.low_subsample_2_bn = nn.BatchNorm2d(512*4)            

        #dataset specific layers        
        self.low_last_conv = nn.Conv2d(512*4, self.output_channel, kernel_size=3, stride=1)
        self.low_average_pool = nn.AvgPool2d(4)
        self.space_final = nn.Linear(1000,self.output_channel)#add fc layer to better transfer
        #self.space_final_dropout = nn.Dropout(self.dropout_ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self,x):
        x = x.view(-1,3,224,224)
        res_out = self.basenet(x)#feed into basenet

        low_level_feature = res_out[0]#getting low and features
        
        #Doing frame subtraction in a video
        fake_batch_size = low_level_feature.size()[0]
        tmp_low_a = low_level_feature[0:fake_batch_size-1,:,:,:]#pytorch NOT include the second edge
        tmp_low_b = low_level_feature[1:fake_batch_size,:,:,:]

        low_level_subtraction = tmp_low_b - tmp_low_a#the subtraction result

        low_level_subtraction_list = []#using torch.cat. delete the invalid data
        
        for i in range(int(fake_batch_size/self.frames_per_video)):
            start_index = i * self.frames_per_video
            end_index = start_index + self.frames_per_video - 1
            #the length of the list is the real batch of video
            #every item in the list should be the size of 14 * C * H * W
            low_level_subtraction_list.append(low_level_subtraction[start_index:end_index,:,:,:])
            
        #size realbatch * 14 * C *H *W
        low_level_subtraction_valid = torch.stack(low_level_subtraction_list) #stack 是创造一个新的axis cat是在一个已经存在的axis上进行的，这是两个函数的区别

        #view
        assert low_level_subtraction_valid.size()[-1] == 28
        assert low_level_subtraction_valid.size()[-3] == 512

        low_level_subtraction_valid = low_level_subtraction_valid.view(-1,512,28,28)

        #conv the valid subtraction
        valid_low_sub_conv = self.sub_conv_low(low_level_subtraction_valid)

        valid_low_sub_conv = valid_low_sub_conv.view(-1,14,512,28,28)

        #Doing sum
        #low_sum_x = torch.sum(valid_low_sub_conv,-1,keepdim=False)#keep x B*14*512*28(H)
        #low_sum_y = torch.sum(valid_low_sub_conv,-2,keepdim=False)#keep y B*14*512*28(W)

        #Doing max
        low_max_x = torch.max(valid_low_sub_conv,-1,keepdim=False)[0]
        low_max_y = torch.max(valid_low_sub_conv,-2,keepdim=False)[0]
        low_sum_x = low_max_x
        low_sum_y = low_max_y
        

        #new to 2D
        low_2d = []
        for i in range(self.frames_per_video-1):
            low_2d.append(low_sum_x[:,i,:,:].unsqueeze(-1))
            low_2d.append(low_sum_y[:,i,:,:].unsqueeze(-1))
        low_2d = torch.cat(low_2d,dim=-1)#B*512*28*28

  
        #low level branch
        low_2d = self.low_layer_node_1(low_2d)
        low_2d = self.low_subsample_1(low_2d)
        low_2d = self.low_subsample_1_bn(low_2d)
        low_2d = self.relu(low_2d)
        low_2d = self.low_layer_node_2(low_2d)
        low_2d = self.low_subsample_2(low_2d)
        low_2d = self.low_subsample_2_bn(low_2d)
        low_2d = self.relu(low_2d)
        low_2d = self.low_last_conv(low_2d)
        low_2d = self.relu(low_2d)
        low_2d = self.low_average_pool(low_2d) #batch size = fake_batch_size / frames_per_video

        #space (B*15)*1000
        space_out = self.space_final(self.relu(res_out[2]))
        #space_out = self.space_final_dropout(space_out)#似乎有问题？？？？？
        space_out = space_out.contiguous().view(-1,self.frames_per_video,self.output_channel).contiguous()
        space_out = space_out.mean(-2)
        return low_2d.squeeze_(-1).squeeze_(-1), space_out       
        
        
