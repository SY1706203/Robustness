from topology_attack import PGDAttack
import torch


def attack_model(recmodel, adj_matrix, perturbations, path, flag, users, posItems, negItems, num_users,
                 use_saved_modified_adj, device):
    model = PGDAttack(model=recmodel, nnodes=adj_matrix.shape[0], loss_type='CE', device=device)

    model = model.to(device)
    print("attack light-GCN model")
    if not use_saved_modified_adj:
        model.attack(adj_matrix, perturbations, users, posItems, negItems, num_users, path, flag)
        modified_adj = model.modified_adj
    else:
        modified_adj = torch.load(path.format(flag))
        modified_adj = modified_adj.to(device)
    print("modified adjacency is same as original adjacency: ", (modified_adj == adj_matrix).all())
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
    print("there are edges between user-user and item-item: ",
          modified_adj[:num_users, :num_users].sum() + modified_adj[num_users:, num_users:].sum() > 0.5)
    return modified_adj


def attack_randomly(recmodel, adj_matrix, perturbations, path, flag, users, posItems, negItems, num_users,
                    use_saved_modified_adj, device):
    """
    if use_saved_modified_adj:
        pgd_adj = torch.load(path.format(flag))
    else:
        pgd_adj = attack_model(recmodel, adj_matrix, perturbations, path, flag, users, posItems, negItems, num_users,
                               use_saved_modified_adj, device)
    """

    num_modified_edges = 186669  # (pgd_adj != adj_matrix).sum().detach().cpu().numpy() // 2
    modification_mask = torch.FloatTensor(num_users, adj_matrix.size()[0] - num_users).uniform_() \
                        <= num_modified_edges / (num_users * (adj_matrix.size()[0] - num_users))
    modification_mask = modification_mask.to(device)
    modified_adj = adj_matrix.detach().clone().bool().to(device)
    modified_adj[:num_users, num_users:] = modified_adj[:num_users, num_users:] ^ modification_mask
    modified_adj[num_users:, :num_users] = modified_adj[num_users:, :num_users] ^ modification_mask.T

    modified_adj[num_users, num_users] = False

    modified_adj = modified_adj.float()
    print("{} edges are modified in modified adj matrix.: ".format((modified_adj != adj_matrix).sum().
                                                                   detach().cpu().numpy()))

    return modified_adj
