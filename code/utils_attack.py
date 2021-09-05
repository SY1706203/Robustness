from topology_attack import PGDAttack
import torch
import os
import lightgcn


def attack_model(recmodel, adj_matrix, perturbations, path, ids, flag, users, posItems, negItems, num_users, device):
    model = PGDAttack(model=recmodel, nnodes=adj_matrix.shape[0], loss_type='CE', device=device)

    model = model.to(device)
    print("attack light-GCN model")
    if not os.path.exists(path.format(ids[flag])):
        model.attack(adj_matrix, perturbations, users, posItems, negItems, num_users, path, ids, flag)
        modified_adj = model.modified_adj
    else:
        print('load matrix from disc...')
        modified_adj = torch.load(path.format(ids[flag]))
        modified_adj = modified_adj.to(device)
    try:
        print("modified adjacency is same as original adjacency: ", (modified_adj == adj_matrix).all())
    except AttributeError:
        print("adjacency is not modified. Check your perturbation and make sure 0 isn't assigned.")
    print("{} edges are in the adjacancy matrix".format(adj_matrix.sum()))
    print("{} edges are in the modified adjacancy matrix".format(modified_adj.sum()))
    print("{} edges are modified in modified adj matrix.".format((modified_adj != adj_matrix).sum().
                                                                 detach().cpu().numpy()))
    print("{} edges are modified in upper triangular matrix of modified adj.".
          format((torch.triu(modified_adj, diagonal=1) != torch.triu(adj_matrix, diagonal=1)).sum().
                 detach().cpu().numpy()))
    print("{} edges are modified in lower triangular matrix of modified adj.".
          format((torch.tril(modified_adj, diagonal=-1) != torch.tril(adj_matrix, diagonal=-1)).sum().
                 detach().cpu().numpy()))
    print("there are edges between user-user and item-item in modified adj matrix: ",
          modified_adj[:num_users, :num_users].sum() + modified_adj[num_users:, num_users:].sum() > 0.5)
    return modified_adj


def attack_randomly(recmodel, adj_matrix, perturbations, path, ids, flag, users, posItems, negItems, num_users, device):
    pgd_adj = attack_model(recmodel, adj_matrix, perturbations, path, ids, flag, users, posItems, negItems, num_users,
                           device)

    num_modified_edges = (pgd_adj != adj_matrix).sum().detach().cpu().numpy() // 2
    print("{} of edges are modified in low/upper triangular matrix from PGD".format(num_modified_edges))
    modification_mask = torch.FloatTensor(num_users, adj_matrix.size()[0] - num_users).uniform_() \
                        <= num_modified_edges / (num_users * (adj_matrix.size()[0] - num_users))
    modification_mask = modification_mask.to(device)
    modified_adj = adj_matrix.detach().clone().bool().to(device)
    modified_adj[:num_users, num_users:] = modified_adj[:num_users, num_users:] ^ modification_mask
    modified_adj[num_users:, :num_users] = modified_adj[num_users:, :num_users] ^ modification_mask.T

    modified_adj[num_users, num_users] = False

    modified_adj = modified_adj.float()
    print("{} edges are modified in randomly modified adj matrix.: ".format((modified_adj != adj_matrix).sum().
                                                                   detach().cpu().numpy()))

    return modified_adj


def fit_lightGCN(device, adj, users, posItems, negItems, modified=True):
    Recmodel = lightgcn.LightGCN(device)
    Recmodel = Recmodel.to(device)
    if modified:
        print("training model on modified adj matrix..")
    else:
        print("training model on original adj matrix..")
    Recmodel.fit(adj, users, posItems, negItems)
    print("finished!")

    return Recmodel
