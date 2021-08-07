import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from topology_attack import PGDAttack
import argparse
import register
from register import dataset
import world
import Procedure
import utils
import lightgcn
import Procedure

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1-keep probability).')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['MOOC'],
                    help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.5, help='perturbation rate')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# TODO: move this param to parser
train_groc = True

adj = dataset.getSparseGraph()
adj = torch.FloatTensor(adj.todense()).to(device)

# perturbations = int(args.ptb_rate * ((dataset.trainDataSize+dataset.testDataSize)//2))
perturbations_a = int(args.ptb_rate * (adj.sum() // 2))
perturbations_b = int(args.ptb_rate * (adj.sum() // 2.5))
print("two perturbations are same: ", perturbations_a == perturbations_b)


users, posItems, negItems = utils.getTrainSet(dataset)

# Setup Victim Model
Recmodel = lightgcn.LightGCN(device)
Recmodel = Recmodel.to(device)

num_users = Recmodel.num_users
# adj=adj.to(device)
print("training original model...")
Recmodel.fit(adj, users, posItems, negItems)
print("finished!")

print("original model performance:")
print("===========================")
Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
print("===========================")
# Setup Attack Model


def attack_model(recmodel, adj_matrix, perturbations, train_groc_):
    model = PGDAttack(model=recmodel, nnodes=adj_matrix.shape[0], loss_type='CE', device=device)

    model = model.to(device)
    print("attack light-GCN model")
    model.attack(adj_matrix, perturbations, users, posItems, negItems, num_users)

    modified_adj = model.modified_adj
    print("modified adjacency is same as original adjacency: ", (modified_adj == adj_matrix).all())

    Recmodel_ = lightgcn.LightGCN(device)
    Recmodel_ = Recmodel_.to(device)
    # adj=adj.to(device)

    if not train_groc_:
        print("train the model with modified adjacency matrix")
        Recmodel_.fit(modified_adj, users, posItems, negItems)

        Procedure.Test(dataset, Recmodel_, 1, utils.normalize_adj_tensor(modified_adj), None, 0)
    return modified_adj


def get_embed_groc(recmodel, modified_adj, users_, poss):
    (users_emb, pos_emb, _, _) = recmodel.getEmbedding(modified_adj, users_.long(), poss.long())
    # pos_emb_old=pos_emb
    users_emb = nn.functional.normalize(users_emb, dim=1)
    pos_emb = nn.functional.normalize(pos_emb, dim=1)

    mlp = utils.MLP(users_emb.size(-1), users_emb.size(-1))
    mlp = mlp.to(device)

    users_emb = mlp(users_emb)
    pos_emb = mlp(pos_emb)

    return torch.cat([users_emb, pos_emb])


def groc_loss_vec(recmodel, modified_adj_a, modified_adj_b, users_, poss):
    all_emb_a = get_embed_groc(recmodel, modified_adj_a, users_, poss)
    all_emb_b = get_embed_groc(recmodel, modified_adj_b, users_, poss)

    contrastive_similarity = torch.exp(torch.diag(torch.matmul(all_emb_a, all_emb_b.t().contiguous())) / recmodel.T)
    # contrastive_similarity size： [batch_size,]
    self_neg_similarity_matrix = torch.matmul(all_emb_a, all_emb_a.t().contiguous())  # z1 * z1
    contrastive_neg_similarity_matrix = torch.matmul(all_emb_a, all_emb_b.t().contiguous())  # z1 * z2
    # self_neg_contrastive_similarity_matrix size： [batch_size, batch_size]
    
    # mask diagonal
    mask = torch.eye(all_emb_b.size(0), all_emb_b.size(0)).bool().to(device)
    # tensor mask with diagonal all True others all False
    self_neg_similarity_matrix.masked_fill_(mask, 0)
    contrastive_neg_similarity_matrix.masked_fill_(mask, 0)
    # concatenates z1 * z1 with 0-diagonal and z1 * z2 with 0-diagonal in row
    # we mask 2 diagonal out because we don't want to get the similarity of an embedding with itself
    neg_contrastive_similarity_matrix = \
        torch.cat([self_neg_similarity_matrix, contrastive_neg_similarity_matrix], -1)
    # sum the matrix up by row
    neg_contrastive_similarity = torch.sum(torch.exp(neg_contrastive_similarity_matrix) / recmodel.T, 1)

    loss_vec = -torch.log(contrastive_similarity / contrastive_similarity + neg_contrastive_similarity)

    return loss_vec


def groc_loss(ori_model, modified_adj_a, modified_adj_b, users_, poss):
    loss_vec_a = groc_loss_vec(ori_model, modified_adj_a, modified_adj_b, users_, poss)
    loss_vec_b = groc_loss_vec(ori_model, modified_adj_b, modified_adj_a, users_, poss)

    return torch.sum(torch.add(loss_vec_a, loss_vec_b)) / (2 * loss_vec_a.size(0))


def groc_train(train_groc_, ori_model):
    modified_adj_a = attack_model(Recmodel, adj, perturbations_a, train_groc_)
    modified_adj_b = attack_model(Recmodel, adj, perturbations_b, train_groc_)

    ori_model.train()

    optimizer = optim.Adam(ori_model.parameters(), lr=ori_model.lr, weight_decay=ori_model.weight_decay)
    for i in range(100):
        optimizer.zero_grad()
        users_ = users.to(ori_model.device)
        posItems_ = posItems.to(ori_model.device)
        negItems_ = negItems.to(ori_model.device)
        users_, posItems_, negItems_ = utils.shuffle(users_, posItems_, negItems_)
        total_batch = len(users) // 2048 + 1
        aver_loss = 0.
        for (batch_i, (batch_users, batch_pos, _)) \
                in enumerate(utils.minibatch(users_, posItems_, negItems_, batch_size=2048)):
            loss = groc_loss(ori_model, modified_adj_a, modified_adj_b, batch_users, batch_pos)

            loss.backward()
            optimizer.step()

            aver_loss += loss.cpu().item()
        aver_loss = aver_loss / total_batch
        if i % 10 == 0:
            print("GROC Loss: ", aver_loss)


# TODO: optimize the code in right file!

if train_groc:
    groc_train(train_groc, Recmodel)
    print("original model performance after GROC learning:")
    print("===========================")
    Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
    print("===========================")
else:
    _ = attack_model(Recmodel, adj, perturbations_a, train_groc)


