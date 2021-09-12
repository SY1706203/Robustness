'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon, ml-1m]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--groc_epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--modified_adj_id', type=list, default=0,
                        help='select adj matrix from modified adj matrix ids')
    parser.add_argument('--train_groc', type=bool, default=False, help='control if train the groc')
    parser.add_argument('--loss_weight_bpr', type=float, default=0.9, help='control loss form')
    parser.add_argument('--modified_models_id', type=int, default=0,
                        help='select model matrix from modified model matrix ids')
    parser.add_argument('--T_groc', type=int, default=0.7, help='param temperature for GROC')
    parser.add_argument('--embedding_attack', type=bool, default=False, help='PDG attack and evaluate')
    parser.add_argument('--pdg_attack', type=bool, default=False, help='PDG attack and evaluate')
    parser.add_argument('--random_perturb', type=bool, default=False, help='perturb adj randomly and compare to PGD')

    return parser.parse_args()
