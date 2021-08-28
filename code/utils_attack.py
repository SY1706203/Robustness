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
    return modified_adj
