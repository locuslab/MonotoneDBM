from util import *
from train_and_eval import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

num_classes=20

MON_DEFAULTS = {
    'alpha': 0.125,
    'tol': 1e-2,
    'max_iter': 50
}

sizes = [(num_classes, 28, 28, 1),
         (40, 14, 14, 10,),
         (80, 7, 7, 20),
         (10, 1)]

kernels = np.array([[3, 0, 0, 0],
                    [3, 3, 0, 0],
                    [3, 3, 3, 0],
                    [0, 0, 1, 1]])

model = ConvDeqCrf(splittingMethod=MONForwardBackwardSplitting,
                   sizes=sizes,
                   kernels=kernels,
                   data_shape=(28, 28),
                   MON_DEFAULTS=MON_DEFAULTS,
                   m=0.01)

model = cuda(model)
trainLoader, testLoader = mnist_inference_loaders(128, test_batch_size=400, num_classes=num_classes)

max_lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=max_lr)
max_epochs = 40
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs * len(trainLoader), eta_min=1e-6)

tune_mrf(train_obs_level=0.4, test_obs_level=0.4, beta=0.9999, train_step=0, trainLoader=trainLoader,
         testLoader=testLoader, model=model, optimizer=optimizer, cuda=cuda, scheduler=None, epochs=max_epochs,
         use_classification=True, use_reconstruction=True, num_classes=num_classes, tune_alpha=True, clf_weight=0.5)
