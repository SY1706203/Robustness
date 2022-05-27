import torch
import numpy as np
import argparse
import os
import random
import lightgcn
import ngcf_ori
from register import dataset
from tqdm import tqdm
from topology_attack import PGDAttack
import utils
from utils_attack import attack_model, attack_randomly, attack_embedding
import Procedure
from scipy.sparse import csc_matrix
from groc_loss import GROC_loss
from adj_generation_LP import links2subgraphs, sample_neg, generate_node2vec_embeddings, subgraph_extraction_labeling
from adj_constraint import Nettack


parser = argparse.ArgumentParser()
parser.add_argument('--seed',                           type=int,   default=15,                                                                                                                                                  help='Random seed.')
parser.add_argument('--warmup_steps',                   type=int,   default=10000,                                                                                                                                               help='Warm up steps for scheduler.')
parser.add_argument('--batch_size',                     type=int,   default=2048,                                                                                                                                                help='BS.')
parser.add_argument('--groc_batch_size',                type=int,   default=10,                                                                                                                                                help='BS.')
parser.add_argument('--groc_epochs',                    type=int,   default=100,                                                                                                                                                 help='Number of epochs to train.')
parser.add_argument('--lr',                             type=float, default=0.001,                                                                                                                                                help='Initial learning rate.')
parser.add_argument('--weight_decay',                   type=float, default=5e-4,                                                                                                                                                help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden',                         type=int,   default=16,                                                                                                                                                  help='Number of hidden units.')
parser.add_argument('--dropout',                        type=float, default=0.5,                                                                                                                                                 help='Dropout rate (1-keep probability).')
parser.add_argument('--train_groc',                     type=bool,  default=False,                                                                                                                                               help='control if train the groc')
parser.add_argument('--pgd_attack',                     type=bool,  default=False,                                                                                                                                               help='PGD attack and evaluate')
parser.add_argument('--embedding_attack',               type=bool,  default=False,                                                                                                                                               help='PGD attack and evaluate')
parser.add_argument('--random_perturb',                 type=bool,  default=False,                                                                                                                                               help='perturb adj randomly and compare to PGD')
parser.add_argument('--groc_with_bpr',                  type=bool,  default=False,                                                                                                                                               help='train a pre-trained GCN on GROC loss')
parser.add_argument('--groc_rdm_adj_attack',            type=bool,  default=False,                                                                                                                                               help='train a pre-trained GCN on GROC loss')
parser.add_argument('--groc_embed_mask',                type=bool,  default=False,                                                                                                                                               help='train a pre-trained GCN on GROC loss')
parser.add_argument('--gcl_with_bpr',                   type=bool,  default=False,                                                                                                                                               help='train a pre-trained GCN on GROC loss')
parser.add_argument('--use_scheduler',                  type=bool,  default=False,                                                                                                                                               help='Use scheduler for learning rate decay')
parser.add_argument('--use_IntegratedGradient',         type=bool,  default=False,                                                                                                                                                help='Use scheduler for learning rate decay')
parser.add_argument('--groc_with_bpr_cat',              type=bool,  default=False,                                                                                                                                                help='Use scheduler for learning rate decay')
parser.add_argument('--use_groc_pgd',                   type=bool,  default=False,                                                                                                                                                help='Use scheduler for learning rate decay')
parser.add_argument('--loss_weight_bpr',                type=float, default=0.9,                                                                                                                                                 help='train loss with learnable weight between 2 losses')
parser.add_argument('--dataset',                        type=str,   default='citeseer',                                                                                                             choices=['MOOC'],            help='dataset')
parser.add_argument('--T_groc',                         type=float, default=0.7,                                                                                                                                                 help='param temperature for GROC')
parser.add_argument('--ptb_rate',                       type=float, default=0.5,                                                                                                                                                 help='perturbation rate')
parser.add_argument('--model',                          type=str,   default='PGD',                                                                                                                  choices=['PGD', 'min-max'],  help='model variant')
parser.add_argument('--embed_attack_method',            type=str,   default='Gradient',                                                                                                             choices=['Gradient', 'rdm'], help='model variant')
parser.add_argument('--path_modified_adj',              type=str,   default=os.path.abspath(os.path.dirname(os.getcwd())) + '/data/modified_adj_{}.pt',                                                                          help='path where modified adj matrix are saved')
parser.add_argument('--modified_adj_name',              type=list,  default=['a_02', 'a_04', 'a_06', 'a_08', 'a_1', 'a_12', 'a_14', 'a_16', 'a_18', 'a_2'],                                                                      help='we attack adj twice for GROC training so we will have 2 modified adj matrix. In order to distinguish them we set a flag to save them independently')
parser.add_argument('--modified_adj_name_with_rdm_ptb_a', type=list,  default=['a_02_w_r', 'a_04_w_r', 'a_06_w_r', 'a_08_w_r', 'a_1_w_r', 'a_12_w_r', 'a_14_w_r', 'a_16_w_r', 'a_18_w_r', 'a_2_w_r'],                              help='we attack adj twice for GROC training, 1st random 2nd PGD.')
parser.add_argument('--modified_adj_name_with_rdm_ptb_b', type=list,  default=['a_02_w_r_b', 'a_04_w_r_b', 'a_06_w_r_b', 'a_08_w_r_b', 'a_1_w_r_b', 'a_12_w_r_b', 'a_14_w_r_b', 'a_16_w_r_b', 'a_18_w_r_b', 'a_2_w_r_b'],                             help='we attack adj twice for GROC training, 1st random 2nd PGD.')
parser.add_argument('--modified_adj_name_with_masked_M_a', type=list,  default=['a_02_mM_a', 'a_04_mM_a', 'a_06_mM_a', 'a_08_mM_a', 'a_1_mM_a', 'a_12_mM_a', 'a_14_mM_a', 'a_16_mM_a', 'a_18_mM_a', 'a_2_mM_a'],                              help='masked_M indicates masked model(embedding mask)')
parser.add_argument('--modified_adj_name_with_masked_M_b', type=list,  default=['a_02_mM_b', 'a_04_mM_b', 'a_06_mM_b', 'a_08_mM_b', 'a_1_mM_b', 'a_12_mM_b', 'a_14_mM_b', 'a_16_mM_b', 'a_18_mM_b', 'a_2_mM_b'],                             help='masked_M indicates masked model(embedding mask)')
parser.add_argument('--mask_prob_list',              type=list,  default=[0.1, 0.2, 0.3, 0.4],                              help='we attack adj twice for GROC training, 1st random 2nd PGD.')
parser.add_argument('--mask_prob_idx',              type=int,  default=1,                              help='we attack adj twice for GROC training, 1st random 2nd PGD.')
parser.add_argument('--perturb_strength_list',          type=list,  default=[10, 5, 3.33, 2.5, 2, 1.67, 1.42, 1.25, 1.11, 1],                                                                                                    help='2 perturb strength for 2 PGD attacks')
parser.add_argument('--modified_adj_id',                type=int,   default=0,                                                                                                                                                   help='select adj matrix from modified adj matrix ids')
parser.add_argument('--masked_model_a_id',                type=int,   default=2,                                                                                                                                                   help='select adj matrix from modified adj matrix ids')
parser.add_argument('--masked_model_b_id',                type=int,   default=1,                                                                                                                                                   help='select adj matrix from modified adj matrix ids')
parser.add_argument('--path_modified_models',           type=str,   default=os.path.abspath(os.path.dirname(os.getcwd())) + '/data/modified_model_{}.pt',                                                                        help='path where modified model is saved')
parser.add_argument('--modified_models_name',           type=list,  default=['02', '04', '06', '08', '1', '12', '14', '16', '18', '2'],                                                                                          help='list of flags for modified models')
parser.add_argument('--eps',                            type=list,  default=[0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2],                                                                                                      help='attack restriction eps for embedding attack')
parser.add_argument('--modified_models_id',             type=int,   default=0,                                                                                                                                                   help='select model matrix from modified model matrix ids')
parser.add_argument('--mask_prob_1',                    type=float,   default=0.3,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--mask_prob_2',                    type=float,   default=0.4,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--insert_prob_1',                    type=float,   default=0.004,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--insert_prob_2',                    type=float,   default=0.004,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--remove_prob_1',                    type=float,   default=0.2,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--remove_prob_2',                    type=float,   default=0.4,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--generate_perturb_adj',         type=bool,   default=True,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--test_ratio',                   type=float,   default=0.2,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--max_train_num',                   type=int,   default=200,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--use_embedding',                   type=bool,   default=False,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--hop',                           type=int,   default=1,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--no_parallel',                     type=bool,   default=True,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--max_nodes_per_hop',                 type=int,   default=20,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--node_percentage_list',                 type=list,   default=[0.25, 0.5, 0.75, 1],                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--node_percentage_list_index',         type=int,   default=0,                                                                                                                                                   help='mask embedding of users/items of GCN')
parser.add_argument('--model_ngcf',                         type=bool,   default=False,                                                                                                                                                   help='mask embedding of users/items of GCN')

args = parser.parse_args()

print("=================================================")
print("All parameters in args")
print(args)
print("=================================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

net = dataset.getSparseGraph()
net = csc_matrix(net)

adj = torch.FloatTensor(net.todense()).to(device)
rowsum = adj.sum(1)
r_inv = rowsum.pow(-1 / 2).flatten()
r_inv[torch.isinf(r_inv)] = 0.
d_mtr = torch.diag(r_inv).to_sparse().to(device)

perturbations = int(args.ptb_rate * (adj.sum() // args.perturb_strength_list[args.modified_adj_id]))
# perturbations = int(args.ptb_rate * ((dataset.trainDataSize+dataset.testDataSize)//2))

# adj = torch.FloatTensor(adj).to(device)

users, posItems, negItems = utils.getTrainSet(dataset)
# users = users[:30]
# posItems = posItems[:30]
# negItems = negItems[:30]
data_len = len(users)

# Setup and fit origin Model

Recmodel = lightgcn.LightGCN(device)
Recmodel = Recmodel.to(device)
# Recmodel.fit(adj, users, posItems, negItems)

num_users = Recmodel.num_users
num_items = Recmodel.num_items
adj_shape = num_users + num_items

if args.model_ngcf:
    print("train model NGCF")
    print("=================================================")

    pgd_model = PGDAttack(model=Recmodel, nnodes=adj.size(1), device=device)
    pgd_model = pgd_model.to(device)

    Recmodel = ngcf_ori.NGCF(num_users, num_items, device)
    Recmodel = Recmodel.to(device)

    groc = GROC_loss(Recmodel, adj, d_mtr, args, pgd_model)
    groc.groc_train_with_bpr(data_len, users, posItems, negItems, perturbations)

    print("save model")
    torch.save(Recmodel.state_dict(), os.path.abspath(os.path.dirname(os.getcwd())) +
               '/data/NGCF_after_GROC_{}.pt'.format(args.loss_weight_bpr))

    print("===========================")

    print("original model performance on original adjacency matrix:")
    print("===========================")
    Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
    print("===========================")

    print("ori model performance after GROC learning on modified adjacency matrix A:")
    print("===========================")
    modified_adj_a = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                  args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
    Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(modified_adj_a), None, 0)

    '''
    # data_prepocessing
    user_fe = torch.load('C:/tmp/user_feature_tensor.pt')
    item_fe = torch.load('C:/tmp/item_feature_tensor.pt')
    num_users = user_fe.size()[0]
    num_items = item_fe.size()[0]

    if user_fe.size()[1] > item_fe.size()[1]:
        target = torch.zeros(item_fe.size()[0], user_fe.size()[1])
        target[:, :item_fe.size()[1]] = item_fe
        feature_dim = user_fe.size()[1]

        feature = torch.cat((user_fe, target), 1)
    else:
        target = torch.zeros(user_fe.size()[0], item_fe.size()[1])
        target[:, :user_fe.size()[1]] = user_fe
        feature_dim = item_fe.size()[1]

        feature = torch.cat((target, item_fe), 1)

    Recmodel = ngcf_ori.NGCF(num_users, num_items, feature_dim, device)
    Recmodel = Recmodel.to(device)
    '''


if args.generate_perturb_adj:
    net = csc_matrix(net)
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net, args.test_ratio, max_train_num=args.max_train_num
    )

    A = net.copy()  # the observed network
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

    node_information = None
    model = utils.Fake_model()
    nettack = Nettack(model=model, nnodes=adj.size(1), device=device)
    # if args.use_embedding:
    #     embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    #     node_information = embeddings
    # for i, j in tqdm(zip(train_pos[0], train_pos[1])):
    #     g, n_labels, n_features = subgraph_extraction_labeling(
    #         (i, j), A, args.hop, args.max_nodes_per_hop, node_information
    #     )
    #     node_list_i = utils.node_list_generation(args, num_users, num_items, i, adj_shape)
    #     node_list_j = utils.node_list_generation(args, num_users, num_items, j, adj_shape)
#
    #     node_i = [i] * len(node_list_i)
    #     node_j = [j] * len(node_list_j)
#
    #     modified_adj = adj.clone().detach()
    #     modified_adj[node_i, node_list_i] = modified_adj[node_i, node_list_i] * (-2) + 1
    #     modified_adj[node_j, node_list_j] = modified_adj[node_j, node_2list_j] * (-2) + 1

    # modified_adj_mask = nettack.get_adj_score_only(features=None, adj=adj, target_node, num_users=num_users,
    #                                              n_perturbations=perturbations)

    # modified_adj = modified_adj * modified_adj_mask

    # train_graphs, test_graphs, max_n_label = links2subgraphs(
    #     A,
    #     train_pos,
    #     train_neg,
    #     test_pos,
    #     test_neg,
    #     args.hop,
    #     args.max_nodes_per_hop,
    #     node_information,
    #     args.no_parallel
    # )

if args.random_perturb:
    print("train model using random perturbation")
    print("=================================================")
    modified_adj = attack_randomly(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                   args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
    try:
        print("modified adjacency is same as original adjacency: ", (modified_adj == adj).all())
    except AttributeError:
        print("adjacency is not modified. Check your perturbation and make sure 0 isn't assigned.")

    Recmodel_ = lightgcn.LightGCN(device)
    Recmodel_ = Recmodel_.to(device)
    Recmodel_.fit(adj, users, posItems, negItems)
    print("evaluate the model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel_, 1, utils.normalize_adj_tensor(modified_adj), None, 0)
    print("=================================================")

if args.train_groc:
    Recmodel = lightgcn.LightGCN(device)
    Recmodel = Recmodel.to(device)
    print("Train GROC loss")
    print("=================================================")
    if args.groc_rdm_adj_attack:
        print("Mode: Random attack + PGD attack")
        rdm_modified_adj_a = attack_randomly(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                             args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
        rdm_modified_adj_b = attack_randomly(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                             args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
        try:
            print("2 random perturbed adj matrix ain't same: ", ~(rdm_modified_adj_a == rdm_modified_adj_b).all())
        except AttributeError:
            print("2 random perturbed adj matrix are same.")

        print("{} edges are different in both random perturbed adj matrix.".
              format((rdm_modified_adj_a != rdm_modified_adj_b).sum().detach().cpu().numpy()))

        groc = GROC_loss(Recmodel, adj, args)
        modified_adj_a, modified_adj_b = groc.attack_adjs(rdm_modified_adj_a, rdm_modified_adj_b, perturbations,
                                                          users, posItems, negItems)
        print("===========================")
        print("Train model_a on modified_adj_a")
        groc.bpr_with_dcl(data_len, modified_adj_a, modified_adj_b, users, posItems, negItems)

        print("original model performance on original adjacency matrix:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
        print("===========================")

        print("trn_model performance after GROC learning on modified adjacency matrix A:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(modified_adj_a), None, 0)
        print("===========================")

        print("trn_model performance after GROC learning on modified adjacency matrix B:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(modified_adj_b), None, 0)

    if args.groc_embed_mask:
        print("Mode: Embedding mask + gradient attack")
        groc = GROC_loss(Recmodel, adj, args)
        groc.groc_train()

        print("original model performance on original adjacency matrix:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
        print("===========================")

        print("ori model performance after GROC learning on modified adjacency matrix A:")
        print("===========================")
        modified_adj_a = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                      args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(modified_adj_a), None, 0)

        print("save model")
        torch.save(Recmodel.state_dict(), os.path.abspath(os.path.dirname(os.getcwd())) + '/data/LightGCN_after_GROC.pt')
        print("===========================")

    if args.groc_with_bpr:
        print("Mode:GROC + BPR")
        pgd_model = PGDAttack(model=Recmodel, nnodes=adj.size(1), device=device)
        pgd_model = pgd_model.to(device)
        Recmodel._is_sparse = True
        groc = GROC_loss(Recmodel, adj, d_mtr, args, pgd_model)
        groc.groc_train_with_bpr(data_len, users, posItems, negItems, perturbations)

        print("save model")
        torch.save(Recmodel.state_dict(), os.path.abspath(os.path.dirname(os.getcwd())) +
                   '/data/LightGCN_after_GROC_{}.pt'.format(args.loss_weight_bpr))
        print("===========================")

        print("original model performance on original adjacency matrix:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
        print("===========================")

        print("ori model performance after GROC learning on modified adjacency matrix A:")
        print("===========================")
        modified_adj_a = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                      args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(modified_adj_a), None, 0)

    if args.gcl_with_bpr:

        def __dropout_x(x, keep_prob):
            size = x.coalesce().size()
            index = x.coalesce().indices().t()
            values = x.coalesce().values()
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            index = index[random_index]
            values = values[random_index] / keep_prob
            g = torch.sparse.FloatTensor(index.t(), values, size)
            return g

        Graph = dataset.getSparseGraph()
        Graph = utils.to_tensor(Graph, device=device)
        Graph1 = __dropout_x(Graph, 0.8).to(device)
        Graph2 = __dropout_x(Graph, 0.8).to(device)

        print("Mode:GCL + BPR")
        groc = GROC_loss(Recmodel, adj, d_mtr, args)
        groc.ori_gcl_train_with_bpr(Graph1, Graph2, data_len, users, posItems, negItems)

        print("original model performance on original adjacency matrix:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
        print("===========================")

        print("ori model performance after GROC learning on modified adjacency matrix A:")
        print("===========================")
        modified_adj_a = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                      args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(modified_adj_a), None, 0)

        print("save model")
        torch.save(Recmodel.state_dict(), os.path.abspath(os.path.dirname(os.getcwd())) + '/data/LightGCN_after_GCL_BPR.pt')
        print("===========================")

    if args.groc_with_bpr_cat:
        print("Mode:GROC + BPR CAT")
        groc = GROC_loss(Recmodel, adj, d_mtr, args)
        groc.groc_train_with_bpr_cat(data_len, users, posItems, negItems)

        print("save model")
        torch.save(Recmodel.state_dict(), os.path.abspath(os.path.dirname(os.getcwd())) +
                   '/data/LightGCN_after_GROC_{}.pt'.format(args.loss_weight_bpr))
        print("===========================")

        print("original model performance on original adjacency matrix:")
        print("===========================")
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(adj), None, 0)
        print("===========================")

        print("ori model performance after GROC learning on modified adjacency matrix A:")
        print("===========================")
        modified_adj_a = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                      args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
        Procedure.Test(dataset, Recmodel, 100, utils.normalize_adj_tensor(modified_adj_a), None, 0)
    print("=================================================")
    print("=================================================")

if args.pgd_attack:
    print("train model with pgd attack")
    print("=================================================")
    # Setup Attack Model
    modified_adj = attack_model(Recmodel, adj, perturbations, args.path_modified_adj, args.modified_adj_name,
                                args.modified_adj_id, users, posItems, negItems, Recmodel.num_users, device)
    Recmodel_ = lightgcn.LightGCN(device)
    Recmodel_ = Recmodel_.to(device)
    Recmodel_.fit(adj, users, posItems, negItems)

    try:
        print("modified adjacency is same as original adjacency: ", (modified_adj == adj).all())
    except AttributeError:
        print("adjacency is not modified. Check your perturbation and make sure 0 isn't assigned.")

    print("evaluate the model with modified adjacency matrix")
    Procedure.Test(dataset, Recmodel_, 1, utils.normalize_adj_tensor(modified_adj), None, 0)
    print("=================================================")

if args.embedding_attack:
    print("train model with embedding adversarial attack")
    print("=================================================")
    fit_model = lightgcn.LightGCN(device)
    modified_model = attack_embedding(fit_model, adj, args.eps[args.modified_models_id],
                                      args.path_modified_models, args.modified_models_name, args.modified_models_id,
                                      users, posItems, negItems, num_users, device)

    fit_model = fit_model.to(device)
    fit_model.fit(adj, users, posItems, negItems)

    print("evaluate the ATTACKED model with original adjacency matrix")
    Procedure.Test(dataset, fit_model, 1, utils.normalize_adj_tensor(adj), None, 0)
    print("=================================================")
