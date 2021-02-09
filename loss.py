import torch
from torch import nn

from constants import DEVICE

loss_fn = nn.BCELoss()
loss_fn_ = nn.MarginRankingLoss(margin=1)
loss_fn__ = nn.BCEWithLogitsLoss()


def calc_loss(discriminator, queries, image_rep, target):
    '''
    queries is a [train_batch_size * F] tensor
    image_rep is a [train_batch_size * F] tensor
    target is a [train_batch_size * 1] tensors
    '''

    # y is the prediction, its shape is [train_batch_size * 1]
    y = discriminator(torch.cat([queries, image_rep], dim=1)).squeeze(1)
    return loss_fn__(y, target.float())


def calc_loss_(queries, pos_image_rep, neg_image_rep):
    batch_size = queries.size(0)
    ones = torch.ones(batch_size).to(DEVICE)
    pos_sim = torch.cosine_similarity(queries, pos_image_rep)
    neg_sim = torch.cosine_similarity(queries, neg_image_rep)
    # print('pos: ', pos_sim)
    # print('neg: ', neg_sim)
    loss = loss_fn_(pos_sim, neg_sim, ones)
    # print(loss)
    return loss


def calc_loss__(queries, image_rep, target):
    y = torch.bmm(queries.view(-1, 1, 512), image_rep.view(-1, 512, 1)).squeeze(2).squeeze(1)
    return loss_fn__(y, target.float())
