import torch
import torch.nn as nn
import torch.nn.functional as F


def ori_gcl_computing(ori_adj, trn_model, gra1, gra2, users, poss, args, device, mask_1=None, mask_2=None, query_groc=None):
    (_, pos_emb, _, _) = trn_model.getEmbedding(ori_adj, users.long(), poss.long())

    (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(), query_groc=query_groc)
    if mask_1 is not None:
        users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1).masked_fill_(mask_1, 0.)
    else:
        users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1)
    (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra2, users.long(), poss.long(), query_groc=query_groc)
    if mask_2 is not None:
        users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1).masked_fill_(mask_2, 0.)
    else:
        users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1)
    users_dot_12 = torch.bmm(users_emb_perturb_1.unsqueeze(1), users_emb_perturb_2.unsqueeze(2)).squeeze(2)
    users_dot_12 /= args.T_groc
    fenzi_12 = torch.exp(users_dot_12).sum(1)

    neg_emb_users_12 = users_emb_perturb_2.unsqueeze(0).repeat(pos_emb.size(0), 1, 1)
    neg_dot_12 = torch.bmm(neg_emb_users_12, users_emb_perturb_1.unsqueeze(2)).squeeze(2)
    neg_dot_12 /= args.T_groc
    neg_dot_12 = torch.exp(neg_dot_12).sum(1)

    mask_11 = get_negative_mask_perturb(users_emb_perturb_1.size(0)).to(device)
    neg_dot_11 = torch.exp(torch.mm(users_emb_perturb_1, users_emb_perturb_1.t()) / args.T_groc)
    neg_dot_11 = neg_dot_11.masked_select(mask_11).view(users_emb_perturb_1.size(0), -1).sum(1)
    loss_perturb_11 = (-torch.log(fenzi_12 / (neg_dot_11 + neg_dot_12))).mean()

    users_dot_21 = torch.bmm(users_emb_perturb_2.unsqueeze(1), users_emb_perturb_1.unsqueeze(2)).squeeze(2)
    users_dot_21 /= args.T_groc
    fenzi_21 = torch.exp(users_dot_21).sum(1)

    neg_emb_users_21 = users_emb_perturb_1.unsqueeze(0).repeat(pos_emb.size(0), 1, 1)
    neg_dot_21 = torch.bmm(neg_emb_users_21, users_emb_perturb_2.unsqueeze(2)).squeeze(2)
    neg_dot_21 /= args.T_groc
    neg_dot_21 = torch.exp(neg_dot_21).sum(1)

    mask_22 = get_negative_mask_perturb(users_emb_perturb_2.size(0)).to(device)
    neg_dot_22 = torch.exp(torch.mm(users_emb_perturb_2, users_emb_perturb_2.t()) / args.T_groc)
    neg_dot_22 = neg_dot_22.masked_select(mask_22).view(users_emb_perturb_2.size(0), -1).sum(1)
    loss_perturb_22 = (-torch.log(fenzi_21 / (neg_dot_22 + neg_dot_21))).mean()

    loss_perturb = loss_perturb_11 + loss_perturb_22

    return loss_perturb


def get_negative_mask_perturb(batch_size):
    negative_mask = torch.ones(batch_size, batch_size).bool()
    for i in range(batch_size):
        negative_mask[i, i] = 0

    return negative_mask
