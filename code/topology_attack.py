import torch
import numpy as np
from tqdm import tqdm
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from base_attack import BaseAttack
import utils


class PGDAttack(BaseAttack):
    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True,
                 attack_features=False, device=None):
        super(PGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'
        self.surrogate._is_sparse = False
        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            assert nnodes is not None, "Please give nnodes="
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes * (nnodes - 1) / 2)))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert True, "Topology Attack does not support attack feature"

        self.complementary = None

    def attack(self, ori_adj, perturbations, users, posItems, negItems, num_users, path, ids, flag):
        victim_model = self.surrogate

        # self.sparse_features=sp.issparse(ori_features)
        # print(sp.issparse(ori_adj))
        ori_adj = utils.to_tensor(ori_adj.cpu(), device=self.device)

        victim_model.eval()
        epochs = 200
        for t in tqdm(range(epochs)):

            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            modified_adj = self.get_modified_adj(ori_adj, num_users)
            adj_norm = utils.normalize_adj_tensor(modified_adj)

            for (batch_i,
                 (batch_users,
                  batch_pos,
                  batch_neg)) in enumerate(utils.minibatch(users,
                                                           posItems,
                                                           negItems,
                                                           batch_size=2048)):
                loss, reg_loss = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg)
                # reg_loss = reg_loss * self.weight_decay
                # loss = loss + reg_loss

                # output=victim_model(adj_norm, users, posItems)
                # loss, reg_loss=victim_model.bpr_loss(adj_norm, users, posItems, negItems)

                adj_grad = torch.autograd.grad(loss, self.adj_changes, retain_graph=True)[0]

                # lr=200/np.sqrt(t+1)
                lr = 200 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
                # print(self.adj_changes) used in perturbation 0.1 log
                self.projection(perturbations)

        self.random_sample(ori_adj, perturbations, users, posItems, negItems, num_users)
        self.modified_adj = self.get_modified_adj(ori_adj, num_users).detach()

        torch.save(self.modified_adj, path.format(ids[flag]))

    def attack_per_batch(self, ori_adj, perturbations, batch_users, batch_pos, batch_neg, num_users):

        victim_model = self.surrogate
        victim_model.eval()

        modified_adj = self.get_modified_adj(ori_adj, num_users)
        adj_norm = utils.normalize_adj_tensor(modified_adj)

        loss, reg_loss = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg)

        adj_grad = torch.autograd.grad(loss, self.adj_changes, retain_graph=True)[0]

        # lr=200/np.sqrt(t+1)
        lr = 200
        self.adj_changes.data.copy_(lr * adj_grad)
        # print(self.adj_changes) used in perturbation 0.1 log
        self.projection(perturbations)

        self.random_sample_batch(victim_model, ori_adj, perturbations, batch_users, batch_pos, batch_neg, num_users)

        adj_modified = self.get_modified_adj(ori_adj, num_users).detach()

        self.adj_changes.data.fill_(0)

        return adj_modified

    def random_sample(self, ori_adj, perturbations, users, posItems, negItems, num_users):
        K = 5
        best_loss = -1000
        victim_model = self.surrogate
        with torch.no_grad():
            s = self.adj_changes.cpu().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if sampled.sum() > perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))

                users = users.to(self.device)
                posItems = posItems.to(self.device)
                negItems = negItems.to(self.device)
                users, posItems, negItems = utils.shuffle(users, posItems, negItems)

                modified_adj = self.get_modified_adj(ori_adj, num_users)
                adj_norm = utils.normalize_adj_tensor(modified_adj)

                loss_total = 0.

                for (batch_i,
                     (batch_users,
                      batch_pos,
                      batch_neg)) in enumerate(utils.minibatch(users,
                                                               posItems,
                                                               negItems,
                                                               batch_size=2048)):
                    loss, reg_loss = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg)

                    # print(loss)
                    loss_total += loss.cpu().item()

                if best_loss < loss_total:
                    best_loss = loss_total
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def random_sample_batch(self, victim_model, ori_adj, perturbations, batch_users, batch_pos, batch_neg, num_users):
        K = 5
        best_loss = -1000
        with torch.no_grad():
            s = self.adj_changes
            for i in range(K):
                sampled = torch.distributions.binomial.Binomial(1, s).sample()

                # print(sampled.sum())
                if sampled.sum() > perturbations:
                    continue
                self.adj_changes.data.copy_(sampled)

                modified_adj = self.get_modified_adj(ori_adj, num_users)
                adj_norm = utils.normalize_adj_tensor(modified_adj)

                loss_total = 0.

                loss, reg_loss = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg)

                # print(loss)
                loss_total += loss.cpu().item()

                if best_loss < loss_total:
                    best_loss = loss_total
                    best_s = sampled
            self.adj_changes.data.copy_(best_s)

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000 * onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
        return loss

    def projection(self, perturbations):
        if torch.clamp(self.adj_changes, 0, 1).sum() > perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj, num_users):
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
            # self.complementary=(1-torch.eye(self.nnodes).to(self.device)-ori_adj)-ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes - 1, col=self.nnodes - 1, offset=0).to(self.device)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj
        # modified_adj=m+ori_adj
        modified_adj[:num_users, :num_users] = 0
        modified_adj[num_users:, num_users:] = 0

        return modified_adj

    def bisection(self, a, b, perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - perturbations

        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            if (func(miu) == 0.0):
                break
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        return miu


class MinMax(PGDAttack):

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None,
                 attack_structure=True, attack_features=False, device='cpu'):
        super(MinMax, self).__init__(model, nnodes, loss_type, feature_shape, attack_structure, attack_features,
                                     device=device)

    def attack(self, ori_adj, perturbations, users, posItems, negItems, num_users, path, ids, flag):
        victim_model = self.surrogate

        # self.sparse_features=sp.issparse(ori_features)
        # ori_adj,ori_features,labels=utils.to_tensor(ori_adj,ori_features,labels,device=self.device)
        ori_adj = utils.to_tensor(ori_adj.cpu(), device=self.device)

        optimizer = optim.Adam(victim_model.parameters(), lr=0.01)

        epochs = 50
        victim_model.eval()
        for t in tqdm(range(epochs)):
            # update victim model
            victim_model.train()
            '''
            modified_adj=self.get_modified_adj(ori_adj)
            adj_norm=utils.normalize_adj_tensor(modified_adj)
            output=victim_model(ori_features,adj_norm)
            loss=self._loss(output[idx_train],labels[idx_train])
            '''
            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            modified_adj = self.get_modified_adj(ori_adj, num_users)
            adj_norm = utils.normalize_adj_tensor(modified_adj)

            for (batch_i,
                 (batch_users,
                  batch_pos,
                  batch_neg)) in enumerate(utils.minibatch(users,
                                                           posItems,
                                                           negItems,
                                                           batch_size=2048)):
                loss, reg_loss = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * 5e-4
                loss = loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # generate pgd attack
            victim_model.eval()
            '''
            modified_adj=self.get_modified_adj(ori_adj)
            adj_norm=utils.normalize_adj_tensor(modified_adj)
            output=victim_model(ori_features,adj_norm)
            loss=self._loss(output[idx_train,labels[idx_train]])
            adj_grad=torch.autograd.grad(loss,self.adj_changes)[0]
            '''
            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            modified_adj = self.get_modified_adj(ori_adj, num_users)
            adj_norm = utils.normalize_adj_tensor(modified_adj)

            for (batch_i,
                 (batch_users,
                  batch_pos,
                  batch_neg)) in enumerate(utils.minibatch(users,
                                                           posItems,
                                                           negItems,
                                                           batch_size=2048)):
                loss, reg_loss = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg)
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            lr = 0.2
            self.adj_changes.data.add_(lr * adj_grad)

            self.projection(perturbations)

        self.random_sample(ori_adj, perturbations, users, posItems, negItems, num_users)
        self.modified_adj = self.get_modified_adj(ori_adj, num_users).detach()


class EmbeddingAttack(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cuda:0'):
        super(EmbeddingAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        if attack_structure:
            self.delta_U = Parameter(torch.FloatTensor(self.surrogate.num_users, self.surrogate.latent_dim))
            self.delta_I = Parameter(torch.FloatTensor(self.surrogate.num_items, self.surrogate.latent_dim))

            self.delta_U.data.fill_(0)
            self.delta_I.data.fill_(0)

    def attack(self, ori_adj, eps, users, posItems, negItems, num_users, path, ids, flag):
        victim_model = self.surrogate

        adj_norm = utils.normalize_adj_tensor(utils.to_tensor(ori_adj.cpu(), device=self.device))

        U_delta_adv = torch.zeros(self.surrogate.num_users, self.surrogate.latent_dim).to(self.device)
        I_delta_adv = torch.zeros(self.surrogate.num_items, self.surrogate.latent_dim).to(self.device)

        victim_model.eval()

        epochs = 100
        for t in tqdm(range(epochs)):
            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            for (batch_i,
                 (batch_users,
                  batch_pos,
                  batch_neg)) in enumerate(utils.minibatch(users,
                                                           posItems,
                                                           negItems,
                                                           batch_size=2048)):
                loss, _ = victim_model.bpr_loss(adj_norm, batch_users, batch_pos, batch_neg, self.delta_U, self.delta_I)

                U_delta_adv += torch.autograd.grad(loss, self.delta_U, retain_graph=True)[0]
                I_delta_adv += torch.autograd.grad(loss, self.delta_I, retain_graph=True)[0]

        U_delta_adv = eps * nn.functional.normalize(U_delta_adv, dim=0)
        I_delta_adv = eps * nn.functional.normalize(I_delta_adv, dim=0)

        victim_model.embedding_user.weight.data.copy_(victim_model.embedding_user.weight + U_delta_adv)
        victim_model.embedding_item.weight.data.copy_(victim_model.embedding_item.weight + I_delta_adv)

        torch.save(victim_model.state_dict(), path.format(ids[flag]))
