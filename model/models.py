import argparse
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
from .senet import se_resnext101_32x4d
from torch.nn import functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

# Define the SE-based Model
class ft_net_SE(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', init_model=None):
        super().__init__()
        model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2
        # model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')

        if stride == 1:
            model_ft.layer4[0].conv2.stride = (1,1)
            model_ft.layer4[0].downsample[0].stride = (1,1)
        if pool == 'avg':
            model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            model_ft.avg_pool = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'avg+max':
            model_ft.avg_pool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.max_pool2 = nn.AdaptiveMaxPool2d((1,1))
        else:
           print('UNKNOW POOLING!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        self.pool  = pool
        # For DenseNet, the feature dim is 2048
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        else:
            self.classifier = ClassBlock(2048, class_num, droprate)
        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            v1 = self.model.avg_pool2(x)
            v2 = self.model.max_pool2(x)
            v = torch.cat((v1,v2), dim = 1)
        else:
            v = self.model.avg_pool(x)
        v = v.view(v.size(0), v.size(1))
        if not self.training:
            return v
        # Convolution layers
        # Pooling and final linear layer
        if self.flag:
            v = self.classifier.add_block(v)
            v = self.new_dropout(v)
            y = self.classifier.classifier(v)
        else:
            y = self.classifier(v)
        return y,v

