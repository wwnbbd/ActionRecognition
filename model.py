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
        self.conv1 = nn.Conv2d(input_num, int(input_num/4), kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(int(input_num/4))
        self.conv2 = nn.Conv2d(int(input_num/4), int(input_num/4), kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(int(input_num/4))
        self.conv3 = nn.Conv2d(int(input_num/4), input_num, kernel_size=1, stride=1)
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
    def __init__(self, frames_per_video = 15, dataset_name = "UCF",dropout_ratio=0.7):
        super(Net, self).__init__()
        self.frames_per_video = frames_per_video
        self.dataset_name = dataset_name
        self.output_channel = datasets[dataset_name]
        self.dropout_ratio = dropout_ratio

        self.relu = nn.ReLU()
        self.basenet = resnet50(pretrained=False)

        #used after substraction
        self.sub_conv_low = Temporal(512)
        self.sub_conv_high = Temporal(1024)

        #Temple branch one low level feature
        self.low_layer_node_1 = Temporal(512)
        self.low_layer_node_2 = Temporal(512)
        self.low_subsample_1 = nn.Conv2d(512, 512*2, kernel_size=1, stride=2)
        self.low_subsample_1_bn = nn.BatchNorm2d(512*2)

        self.low_layer_node_3 = Temporal(512*2)
        self.low_layer_node_4 = Temporal(512*2)
        self.low_subsample_2 = nn.Conv2d(512*2, 512*4, kernel_size=1, stride=2)
        self.low_subsample_2_bn = nn.BatchNorm2d(512*4)            

        #Temple branch two high level feature
        self.high_layer_node_1 = Temporal(1024)
        self.high_layer_node_2 = Temporal(1024)
        self.high_subsample_1 = nn.Conv2d(1024, 1024*2, kernel_size=1, stride=2)
        self.high_subsample_1_bn = nn.BatchNorm2d(1024*2)

        #dataset specific layers
        
        self.low_last_conv = nn.Conv2d(512*4, self.output_channel, kernel_size=3, stride=1)
        self.low_average_pool = nn.AvgPool2d(4)
        self.high_last_conv = nn.Conv2d(1024*2, self.output_channel, kernel_size=3, stride=1)
        self.high_average_pool = nn.AvgPool2d((4,11))
        self.space_final = nn.Linear(2048,self.output_channel)
        self.space_final_dropout = nn.Dropout(self.dropout_ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self,x):
        res_out = self.basenet(x)#feed into basenet

        low_level_feature = res_out[0]#getting low and high level features
        high_level_feature = res_out[1]

        #Doing frame subtraction in a video
        fake_batch_size = low_level_feature.size()[0]
        tmp_low_a = low_level_feature[0:fake_batch_size-1,:,:,:]#pytorch NOT include the second edge
        tmp_low_b = low_level_feature[1:fake_batch_size,:,:,:]
        tmp_high_a = high_level_feature[0:fake_batch_size-1,:,:,:]
        tmp_high_b = high_level_feature[1:fake_batch_size,:,:,:]

        low_level_subtraction = tmp_low_b - tmp_low_a#the subtraction result
        high_level_subtraction = tmp_high_b - tmp_high_a

        low_level_subtraction_list = []#using torch.cat. delete the invalid data
        high_level_subtraction_list = []
        for i in range(int(fake_batch_size/self.frames_per_video)):
            start_index = i * self.frames_per_video
            end_index = start_index + self.frames_per_video - 1
            low_level_subtraction_list.append(low_level_subtraction[start_index:end_index,:,:,:])
            high_level_subtraction_list.append(high_level_subtraction[start_index:end_index,:,:,:])
        low_level_subtraction_valid = torch.cat(low_level_subtraction_list, dim=0).contiguous()
        high_level_subtraction_valid = torch.cat(high_level_subtraction_list, dim=0).contiguous()

        #conv the valid subtraction
        valid_low_sub_conv = self.sub_conv_low(low_level_subtraction_valid)
        valid_high_sub_conv = self.sub_conv_high(high_level_subtraction_valid)

        #Doing sum
        low_sum_x = valid_low_sub_conv.sum(-1).unsqueeze(-1)#W column B*C*H*1
        low_sum_y = valid_low_sub_conv.sum(-2).unsqueeze(-2)#H row B*C*1*W
        high_sum_x = valid_high_sub_conv.sum(-1).unsqueeze(-1)#W
        high_sum_y = valid_high_sub_conv.sum(-2).unsqueeze(-2)#H

        #video to 2D
        low_sum_y = torch.transpose(low_sum_y, -1, -2)
        high_sum_y = torch.transpose(high_sum_y, -1, -2)

        low_2d = torch.cat([low_sum_x, low_sum_y], dim=-1)#input should be square 
        high_2d = torch.cat([high_sum_x, high_sum_y], dim=-1)

        low_2d = low_2d.view(-1,512,28,(self.frames_per_video-1)*2).contiguous()
        high_2d = high_2d.view(-1,1024,14,(self.frames_per_video-1)*2).contiguous()

        #low level branch
        low_2d = self.low_layer_node_1(low_2d)
        low_2d = self.low_layer_node_2(low_2d)
        low_2d = self.low_subsample_1(low_2d)
        low_2d = self.low_subsample_1_bn(low_2d)
        low_2d = self.relu(low_2d)
        low_2d = self.low_layer_node_3(low_2d)
        low_2d = self.low_layer_node_4(low_2d)
        low_2d = self.low_subsample_2(low_2d)
        low_2d = self.low_subsample_2_bn(low_2d)
        low_2d = self.relu(low_2d)
        low_2d = self.low_last_conv(low_2d)
        low_2d = self.relu(low_2d)
        low_2d = self.low_average_pool(low_2d) #batch size = fake_batch_size / frames_per_video

        #high level branch
        high_2d = self.high_layer_node_1(high_2d)
        high_2d = self.high_layer_node_2(high_2d)
        high_2d = self.high_subsample_1(high_2d)
        high_2d = self.high_subsample_1_bn(high_2d)
        high_2d = self.relu(high_2d)
        high_2d = self.high_last_conv(high_2d)
        high_2d = self.relu(high_2d)
        high_2d = self.high_average_pool(high_2d)   #batch size = fake_batch_size / frames_per_video

        #space
        space_out = self.space_final(res_out[2].view(-1,2048))
        space_out = self.space_final_dropout(space_out)
        space_out = space_out.contiguous().view(-1,self.frames_per_video,self.output_channel).contiguous()
        space_out = space_out.mean(-2)
        return low_2d.squeeze_(-1).squeeze_(-1), high_2d.squeeze_(-1).squeeze_(-1), space_out       
        
        
