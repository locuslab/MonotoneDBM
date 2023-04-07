import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

def mnist_inference_loaders(train_batch_size, test_batch_size=None, num_classes=16):
    class BinaryToTensor(object):
        """H x W x C -> C x H x W"""

        def __init__(self, num_classes):
            self.num_classes = num_classes

        def __call__(self, picture):
            # num_classes represents the number of buckets we want, e.g 16
            # when create linspace, the #steps should be num_classes+1
            # boundaries[-1] should be set > 1 (assume the tensor<=1)

            pic_tensor = torchvision.transforms.ToTensor()(picture)
            boundaries = torch.linspace(0, 1, self.num_classes + 1).to(pic_tensor.device)
            boundaries[-1] = 1.1
            res = (torch.bucketize(pic_tensor, boundaries, right=True) - 1).float()
            return res

    if test_batch_size is None:
        test_batch_size = train_batch_size

    trainLoader = torch.utils.data.DataLoader(
        dset.MNIST('../data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       BinaryToTensor(num_classes),
                   ])),
        batch_size=train_batch_size,
        shuffle=True)

    testLoader = torch.utils.data.DataLoader(
        dset.MNIST('../data',
                   train=False,
                   transform=transforms.Compose([
                       BinaryToTensor(num_classes),
                   ])),
        batch_size=test_batch_size,
        shuffle=False)
    return trainLoader, testLoader

def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda(0)
    else:
        return tensor

def expand_args(defaults, kwargs):
    d = defaults.copy()
    for k, v in kwargs.items():
        d[k] = v
    return d
