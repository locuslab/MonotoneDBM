import time

import torch
import torch.nn as nn

from util import *
from deq_model import *
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm




def eval_mrf_generation(model, obs_level, setting, test_loader, num_classes, run_acc, cuda):
    model.eval()
    for idx, (val_X, val_y) in enumerate(test_loader):
        if idx >= random.choice(list(range(50))):
            val_X = val_X[0:40]
            val_y = val_y[0:40]
            break
    orig_X = cuda(val_X)

    val_X = cuda(F.one_hot(val_X.long().squeeze(), num_classes=num_classes).permute(0, 3, 1, 2).float())
    bsz, c, H, W = val_X.shape

    val_obs_idx = cuda(torch.zeros(bsz, 1, H, W).bernoulli_(obs_level))
    mask = val_obs_idx.repeat(1, c, 1, 1)

    with torch.no_grad():
        all_out = model((val_X,), mask=mask)

    val_output_q = all_out[0]
    val_output_q = val_output_q * (1 - mask) + val_X * mask
    val_output_q = val_output_q.argmax(1, keepdim=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 20))

    ax1.imshow(
        make_grid((orig_X * val_obs_idx).float().view(bsz, -1, H, W).data.cpu(), nrow=8,
                  normalize=True).numpy().transpose(1, 2, 0))
    ax2.imshow(
        make_grid(val_output_q.float().view(bsz, -1, H, W).data.cpu(), nrow=8, normalize=True).numpy().transpose(1, 2, 0))
    ax3.imshow(
        make_grid(orig_X.float().view(bsz, -1, H, W).data.cpu(), nrow=8, normalize=True).numpy().transpose(1, 2, 0))
    ax1.set_title(setting)

    plt.show()

    if run_acc:
        total_err = 0
        nProcessed = 0
        tk0 = tqdm(enumerate(test_loader), leave=True)
        for idx, (val_X, val_y) in tk0:
            val_X = cuda(F.one_hot(val_X.long().squeeze(), num_classes=num_classes).permute(0, 3, 1, 2).float())
            val_y = cuda(val_y)

            bsz, c, H, W = val_X.shape

            val_obs_idx = cuda(torch.zeros(bsz, 1, H, W).bernoulli_(obs_level))
            mask = val_obs_idx.repeat(1, c, 1, 1)

            with torch.no_grad():
                all_out = model((val_X,), mask=mask)

            total_err += (all_out[-1].squeeze().max(dim=1)[1] != val_y).sum().item()
            nProcessed += len(val_X)

            tk0.set_description(f"Test")
            tk0.set_postfix(err=total_err / nProcessed,
                            forward=model.mon.forward_steps,
                            backward=model.mon.backward_steps,
                            forward_res=model.mon.forward_res,
                            backward_res=model.mon.backward_res)
        print(total_err / nProcessed)


def get_logits(model, z_tuple, mask, injection):
    masked_z = z_tuple
    masked_z[0] = masked_z[0] * (1 - mask) + mask * injection[0]
    if len(injection) > 1:
        masked_z[-1] = injection[-1][:, :, None, None]
    linear_out = model.linear_module(*masked_z)
    return linear_out

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.):
        # alpha should be a tensor
        super(FocalLoss, self).__init__()
        self.loss = nn.NLLLoss(weight=alpha, reduction='mean')
        self.gamma = gamma

    def forward(self, log_pred, target, mask_1c, num_classes, tau):
        log_pred = log_pred * (tau ** 2)
        log_pred = log_pred.permute(0, 2, 3, 1).reshape(-1, num_classes)[mask_1c.view(-1) == 0]
        target = (target.permute(0, 2, 3, 1).reshape(-1))[mask_1c.view(-1) == 0]
        softmax_p = F.softmax(log_pred, 1)
        focal_weight = (1 - softmax_p) ** self.gamma
        weighted_log_pred = focal_weight * (log_pred - torch.logsumexp(log_pred, 1, keepdim=True))
        return self.loss(weighted_log_pred, target)


def tune_mrf(train_obs_level, test_obs_level, beta, train_step, trainLoader, testLoader, model, optimizer, cuda,
             scheduler=None, epochs=15, use_classification=True, use_reconstruction=False, num_classes=2, tune_alpha=True, clf_weight=0.9):

    model = cuda(model)
    all_losses = []
    for epoch in range(0, epochs):
        nProcessed = 0
        nTrain = len(trainLoader.dataset)
        model.train()
        start = time.time()
        tk0 = tqdm(enumerate(trainLoader), leave=True)
        total_err = 0

        for batch_idx, batch in tk0:
            onehot_data = F.one_hot(batch[0].long().squeeze(dim=1), num_classes=num_classes).permute(0, 3, 1, 2).float()
            data, target, label = cuda(onehot_data), cuda(batch[0].long()), cuda(batch[1])
            bsz, c, h, w = data.shape

            mask_1c = torch.zeros(bsz, 1, h, w).bernoulli_(train_obs_level)
            mask = cuda(mask_1c.repeat(1, c, 1, 1))

            optimizer.zero_grad()

            all_out = model([data, ], mask=mask)
            unobs_target = (target.permute(0, 2, 3, 1).reshape(-1))[mask_1c.view(-1) == 0].long()
            alpha = (1 - beta) / (1 - beta ** torch.bincount(unobs_target.view(-1)))
            log_pred = get_logits(model, all_out, mask, [data, ])
            ce_loss = FocalLoss(alpha=alpha)(log_pred[0], target, mask_1c, num_classes, model.tau)

            if (model.mon.forward_steps == model.mon.max_iter - 2) and tune_alpha:
                run_tune_alpha(model, data, model.mon.alpha / 2, mask=mask)
            classification_loss = torch.tensor(0)
            if use_classification and not use_reconstruction:
                classification_loss = nn.CrossEntropyLoss()(log_pred[-1].squeeze() * (model.clftau ** 2), label.long())
                classification_loss.backward()
                total_err += (all_out[-1].squeeze().max(dim=1)[1] != label).sum().item()
            elif use_reconstruction and not use_classification:
                ce_loss.backward()
            else:
                classification_loss = nn.CrossEntropyLoss()(log_pred[-1].squeeze() * (model.clftau ** 2),
                                                            label.long())
                total_loss = (1 - clf_weight) * ce_loss + clf_weight * classification_loss
                total_loss.backward()
                total_err += (all_out[-1].squeeze().max(dim=1)[1] != label).sum().item()

            nProcessed += len(data)

            all_losses.append(ce_loss.item())
            tk0.set_description(f"Train Epoch {epoch}")
            tk0.set_postfix(step=f"{train_step}",
                            loss=ce_loss.item(),
                            err=total_err / nProcessed,
                            clf=classification_loss.item(),
                            forward=model.mon.forward_steps,
                            backward=model.mon.backward_steps,
                            forward_res=model.mon.forward_res,
                            backward_res=model.mon.backward_res)
            train_step += 1

            optimizer.step()
            if scheduler:
                scheduler.step()

        eval_mrf_generation(model, test_obs_level, f'beta_{beta}_epoch{epoch}', testLoader, num_classes,
                            run_acc=use_classification, cuda=cuda)

        print("Tot train time: {}".format(time.time() - start))