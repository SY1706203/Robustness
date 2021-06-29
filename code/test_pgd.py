import torch
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

parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=15,help='Random seed.')
parser.add_argument('--epochs',type=int,default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr',type=float,default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay',type=float,default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden',type=int,default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout',type=float,default=0.5,
                    help='Dropout rate (1-keep probability).')
parser.add_argument('--dataset',type=str,default='citeseer',choices=['MOOC'],
                    help='dataset')
parser.add_argument('--ptb_rate',type=float,default=0.2,help='perturbation rate')
parser.add_argument('--model',type=str,default='PGD',choices=['PGD','min-max'],help='model variant')

args=parser.parse_args()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device!='cpu':
    torch.cuda.manual_seed(args.seed)


adj=dataset.getSparseGraph()
adj = torch.FloatTensor(adj.todense()).to(device)


#perturbations = int(args.ptb_rate * ((dataset.trainDataSize+dataset.testDataSize)//2))
perturbations = int(args.ptb_rate * (adj.sum()//2))
#print(perturbations)



users, posItems, negItems=utils.getTrainSet(dataset)

# Setup Victim Model
Recmodel = lightgcn.LightGCN(device)
Recmodel=Recmodel.to(device)
#adj=adj.to(device)
Recmodel.fit(adj, users, posItems, negItems)

Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
# Setup Attack Model

model=PGDAttack(model=Recmodel,nnodes=adj.shape[0],loss_type='CE',device=device)

model=model.to(device)

model.attack(adj, perturbations, users, posItems, negItems)

modified_adj=model.modified_adj
print((modified_adj==adj).all())


Recmodel_ = lightgcn.LightGCN(device)
Recmodel_=Recmodel_.to(device)
#adj=adj.to(device)
Recmodel_.fit(modified_adj, users, posItems, negItems)

Procedure.Test(dataset, Recmodel_, 1, utils.normalize_adj_tensor(modified_adj), None, 0)