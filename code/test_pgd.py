import torch
import numpy as np
import argparse
from register import dataset
from utils import getTrainSet, normalize_adj_tensor
from utils_attack import attack_model, attack_randomly
import lightgcn
import Procedure
import os
from groc_loss import GROC_loss


parser = argparse.ArgumentParser()
parser.add_argument('--seed',               type=int,   default=15,                                                                                       help='Random seed.')
parser.add_argument('--warmup_steps',       type=int,   default=100000,                                                                                   help='Warm up steps for scheduler.')
parser.add_argument('--batch_size',         type=int,   default=2048,                                                                                     help='BS.')
parser.add_argument('--epochs',             type=int,   default=200,                                                                                      help='Number of epochs to train.')
parser.add_argument('--lr',                 type=float, default=0.01,                                                                                     help='Initial learning rate.')
parser.add_argument('--weight_decay',       type=float, default=5e-4,                                                                                     help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden',             type=int,   default=16,                                                                                       help='Number of hidden units.')
parser.add_argument('--dropout',            type=float, default=0.5,                                                                                      help='Dropout rate (1-keep probability).')
parser.add_argument('--train_groc',         type=bool,  default=False,                                                                                    help='control if train the groc')
parser.add_argument('--pdg_attack',         type=bool,  default=True,                                                                                     help='PDG attack and evaluate')
parser.add_argument('--random_perturb',     type=bool,  default=False,                                                                                    help='perturb adj randomly and compare to PGD')
parser.add_argument('--dataset',            type=str,   default='citeseer', choices=['MOOC'],                                                             help='dataset')
parser.add_argument('--T_groc',             type=int,   default=0.7,                                                                                      help='param temperature for GROC')
parser.add_argument('--ptb_rate',           type=float, default=0.5,                                                                                      help='perturbation rate')
parser.add_argument('--model',              type=str,   default='PGD',      choices=['PGD', 'min-max'],                                                   help='model variant')
parser.add_argument('--valid_perturbation', type=bool,  default=True,                                                                                     help='perturbation validation')
parser.add_argument('--train_cascade',      type=bool,  default=False,                                                                                    help='train original model first then train model with GROC loss')
parser.add_argument('--path_modified_adj',  type=str,   default=os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/modified_adj_{}.pt', help='path where modified adj matrix are saved')
parser.add_argument('--modified_adj_flag',  type=list, default=['a', 'b'],                                                                                help='we attack adj twice for GROC training so we will have 2 modified adj matrix. In order to distinguish them we set a flag to save them independently')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

adj = dataset.getSparseGraph()
adj = torch.FloatTensor(adj.todense()).to(device)

# perturbations = int(args.ptb_rate * ((dataset.trainDataSize+dataset.testDataSize)//2))
perturbations_a = int(args.ptb_rate * (adj.sum() // 2))
perturbations_b = int(args.ptb_rate * (adj.sum() // 2.5))
print("two perturbations are same: ", perturbations_a == perturbations_b)


users, posItems, negItems = getTrainSet(dataset)
data_len = len(users)
# Setup Victim Model
Recmodel = lightgcn.LightGCN(device)
Recmodel = Recmodel.to(device)

num_users = Recmodel.num_users
# adj=adj.to(device)
if args.random_perturb:
    modified_adj = attack_randomly(Recmodel, adj, perturbations_a, args.path_modified_adj, args.modified_adj_flag[0],
                                   users, posItems, negItems, Recmodel.num_users,
                                   os.path.exists(args.path_modified_adj.format(args.modified_adj_flag)), device)

    print("training original model...")
    Recmodel.fit(adj, users, posItems, negItems)
    print("finished!")

    print("evaluate the original model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel, 1, normalize_adj_tensor(modified_adj), None, 0)

if args.train_cascade or args.valid_perturbation:  # mostly used for validation
    print("training original model...")
    Recmodel.fit(adj, users, posItems, negItems)
    print("finished!")

    print("original model performance:")
    print("===========================")
    Procedure.Test(dataset, Recmodel, 100, normalize_adj_tensor(adj), None, 0)
    print("===========================")

if args.train_groc:
    groc = GROC_loss(Recmodel, args, users, posItems, negItems)
    modified_adj_a, modified_adj_b = groc.groc_train(data_len, adj, perturbations_a, perturbations_b, users)

    if args.train_cascade:
        print("original model performance after GROC learning:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, normalize_adj_tensor(adj), None, 0)
        print("===========================")
    else:  # train BPR + GROC loss together
        print("original model performance after GROC learning on original adjacency matrix:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, normalize_adj_tensor(adj), None, 0)
        print("===========================")

        print("original model performance after GROC learning on modified adjacency matrix A:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, normalize_adj_tensor(modified_adj_a), None, 0)
        print("===========================")

        print("original model performance after GROC learning on  modified adjacency matrix B:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, normalize_adj_tensor(modified_adj_b), None, 0)
        print("===========================")

if args.pdg_attack:
    # Setup Attack Model
    adj_dir = args.path_modified_adj.format(args.modified_adj_flag[0])
    if os.path.exists(adj_dir):
        modified_adj = torch.load(adj_dir).to(device)
    else:
        modified_adj = attack_model(Recmodel, adj, perturbations_a, args.path_modified_adj, args.modified_adj_flag[0],
                                    users, posItems, negItems, Recmodel.num_users, device)
    Recmodel_ = lightgcn.LightGCN(device)
    Recmodel_ = Recmodel_.to(device)
    print("train the model with modified adjacency matrix")
    Recmodel_.fit(modified_adj, users, posItems, negItems)
    print("evaluate the model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel_, 1, normalize_adj_tensor(modified_adj), None, 0)
    if args.valid_perturbation:
        print("evaluate the original model with modified adjacency matrix")
        Procedure.Test(dataset, Recmodel, 1, normalize_adj_tensor(modified_adj), None, 0)
