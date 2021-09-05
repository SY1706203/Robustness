import torch
import numpy as np
import argparse
from register import dataset
from utils import getTrainSet, normalize_adj_tensor
from utils_attack import attack_model, attack_randomly, fit_lightGCN
import Procedure
import os
from groc_loss import GROC_loss


parser = argparse.ArgumentParser()
parser.add_argument('--seed',                    type=int,   default=15,                                                                             help='Random seed.')
parser.add_argument('--warmup_steps',            type=int,   default=100000,                                                                         help='Warm up steps for scheduler.')
parser.add_argument('--batch_size',              type=int,   default=2048,                                                                           help='BS.')
parser.add_argument('--epochs',                  type=int,   default=200,                                                                            help='Number of epochs to train.')
parser.add_argument('--lr',                      type=float, default=0.01,                                                                           help='Initial learning rate.')
parser.add_argument('--weight_decay',            type=float, default=5e-4,                                                                           help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden',                  type=int,   default=16,                                                                             help='Number of hidden units.')
parser.add_argument('--dropout',                 type=float, default=0.5,                                                                            help='Dropout rate (1-keep probability).')
parser.add_argument('--train_groc',              type=bool,  default=False,                                                                          help='control if train the groc')
parser.add_argument('--pdg_attack',              type=bool,  default=False,                                                                          help='PDG attack and evaluate')
parser.add_argument('--random_perturb',          type=bool,  default=False,                                                                          help='perturb adj randomly and compare to PGD')
parser.add_argument('--dataset',                 type=str,   default='citeseer', choices=['MOOC'],                                                   help='dataset')
parser.add_argument('--T_groc',                  type=int,   default=0.7,                                                                            help='param temperature for GROC')
parser.add_argument('--ptb_rate',                type=float, default=0.5,                                                                            help='perturbation rate')
parser.add_argument('--model',                   type=str,   default='PGD',      choices=['PGD', 'min-max'],                                         help='model variant')
parser.add_argument('--path_modified_adj',       type=str,   default=os.path.abspath(os.path.dirname(os.getcwd())) + '/data/modified_adj_{}.pt',     help='path where modified adj matrix are saved')
parser.add_argument('--modified_adj_name',       type=list,  default=['a_02', 'a_04', 'a_06', 'a_08', 'a_1', 'a_12', 'a_14', 'a_16', 'a_18', 'a_2'], help='we attack adj twice for GROC training so we will have 2 modified adj matrix. In order to distinguish them we set a flag to save them independently')
parser.add_argument('--perturb_strength_list',   type=list,  default=[10, 5, 3.33, 2.5, 2, 1.67, 1.42, 1.25, 1.11, 1],                               help='2 perturb strength for 2 PDG attacks')
parser.add_argument('--modified_adj_id',         type=list,  default=[0, 1],                                                                         help='select adj matrix from modified adj matrix ids')


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

adj = dataset.getSparseGraph()
adj = torch.FloatTensor(adj.todense()).to(device)

# perturbations = int(args.ptb_rate * ((dataset.trainDataSize+dataset.testDataSize)//2))
perturbations_a = int(args.ptb_rate * (adj.sum() // args.perturb_strength_list[args.modified_adj_id[0]]))
perturbations_b = int(args.ptb_rate * (adj.sum() // args.perturb_strength_list[args.modified_adj_id[1]]))
print("two perturbations are same: ", perturbations_a == perturbations_b)


users, posItems, negItems = getTrainSet(dataset)
data_len = len(users)
# Setup and fit origin Model

Recmodel = fit_lightGCN(device, adj, users, posItems, negItems, modified=False)

num_users = Recmodel.num_users

if args.random_perturb:
    modified_adj = attack_randomly(Recmodel, adj, perturbations_a, args.path_modified_adj, args.modified_adj_name,
                                   args.modified_adj_id[0], users, posItems, negItems, Recmodel.num_users, device)
    try:
        print("modified adjacency is same as original adjacency: ", (modified_adj == adj).all())
    except AttributeError:
        print("adjacency is not modified. Check your perturbation and make sure 0 isn't assigned.")

    Recmodel_ = fit_lightGCN(device, modified_adj, users, posItems, negItems)
    print("evaluate the model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel_, 1, normalize_adj_tensor(modified_adj), None, 0)
    print("evaluate the model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel, 1, normalize_adj_tensor(modified_adj), None, 0)

if args.train_groc:
    groc = GROC_loss(Recmodel, args, users, posItems, negItems)
    groc.groc_train(data_len, adj, perturbations_a, perturbations_b, users)
    modified_adj_a, modified_adj_b = groc.modified_adj_a, groc.modified_adj_b

    Recmodel_a = fit_lightGCN(device, modified_adj_a, users, posItems, negItems)
    Recmodel_b = fit_lightGCN(device, modified_adj_b, users, posItems, negItems)

    print("original model performance after GROC learning on original adjacency matrix:")
    print("===========================")
    Procedure.Test(dataset, Recmodel, 100, normalize_adj_tensor(adj), None, 0)
    print("===========================")

    print("original model performance after GROC learning on modified adjacency matrix A:")
    print("===========================")
    Procedure.Test(dataset, Recmodel, 100, normalize_adj_tensor(modified_adj_a), None, 0)
    print("===========================")

    print("new model a performance after GROC learning on modified adjacency matrix A:")
    print("===========================")
    Procedure.Test(dataset, Recmodel_a, 100, normalize_adj_tensor(modified_adj_a), None, 0)
    print("===========================")

    print("original model performance after GROC learning on  modified adjacency matrix B:")
    print("===========================")
    Procedure.Test(dataset, Recmodel, 100, normalize_adj_tensor(modified_adj_b), None, 0)
    print("===========================")

    print("new model b performance after GROC learning on modified adjacency matrix B:")
    print("===========================")
    Procedure.Test(dataset, Recmodel_b, 100, normalize_adj_tensor(modified_adj_b), None, 0)
    print("===========================")

if args.pdg_attack:
    # Setup Attack Model
    modified_adj = attack_model(Recmodel, adj, perturbations_a, args.path_modified_adj, args.modified_adj_name,
                                args.modified_adj_id[0], users, posItems, negItems, Recmodel.num_users, device)
    Recmodel_ = fit_lightGCN(device, modified_adj, users, posItems, negItems)

    try:
        print("modified adjacency is same as original adjacency: ", (modified_adj == adj).all())
    except AttributeError:
        print("adjacency is not modified. Check your perturbation and make sure 0 isn't assigned.")

    print("evaluate the model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel_, 1, normalize_adj_tensor(modified_adj), None, 0)

    print("evaluate the original model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel, 1, normalize_adj_tensor(modified_adj), None, 0)
