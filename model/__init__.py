from .models import *
from .optimizers import *
from .resnet import *
from .senet import *

__model_factory = {
    # image classification models
    'resnet50': resnet50,
    'resnet50_fc512': resnet50_fc512,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet152_ibn_a': resnet152_ibn_a,
    'senet154': senet154,
    'se_resnet50': se_resnet50,
    'se_resnet101': se_resnet101,
    'se_resnet152': se_resnet152,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d }

def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*args, **kwargs)
