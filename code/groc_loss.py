import gc
import torch
import torch.nn as nn
import torch.optim as optim
from utils import scheduler_groc
from utils_attack import attack_model
import utils
import torch.nn.functional as F


class GROC_loss(nn.Module):
    def __init__(self, ori_model, ori_adj, args):
        super(GROC_loss, self).__init__()
        self.ori_adj = ori_adj
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ori_model = ori_model
        self.args = args
        self.num_users = self.ori_model.num_users
        self.num_items = self.ori_model.num_items

    def get_embed_groc(self, trn_model, modified_adj, users, items, mask):

        adj_norm = utils.normalize_adj_tensor(modified_adj, sparse=True)
        modified_adj = adj_norm.to(self.device)

        del adj_norm
        gc.collect()  # garbage collection of passed-in tensor

        (users_emb, item_emb, _, _) = trn_model.getEmbedding(modified_adj, users.long(), items.long(), query_groc=True)
        if mask is not None:
            users_emb = nn.functional.normalize(users_emb, dim=1).masked_fill_(mask, 0.)
            item_emb = nn.functional.normalize(item_emb, dim=1).masked_fill_(mask, 0.)
        else:
            users_emb = nn.functional.normalize(users_emb, dim=1)
            item_emb = nn.functional.normalize(item_emb, dim=1)

        return torch.cat([users_emb, item_emb])

    def groc_loss_vec(self, trn_model, modified_adj_a, modified_adj_b, users, items, mask_1, mask_2):
        batch_emb_a = self.get_embed_groc(trn_model, modified_adj_a, users, items, mask_1)
        batch_emb_b = self.get_embed_groc(trn_model, modified_adj_b, users, items, mask_2)

        contrastive_similarity = torch.exp(torch.sum(batch_emb_a * batch_emb_b, dim=-1) / self.args.T_groc)

        # contrastive_similarity size： [batch_size,]
        self_neg_similarity_matrix = torch.matmul(batch_emb_a, batch_emb_a.t().contiguous())  # tau_1(v) * tau_2(v)
        contrastive_neg_similarity_matrix = torch.matmul(batch_emb_a, batch_emb_b.t().contiguous())
        # tau_1(v) * tau2(neg), neg includes v itself, will be masked below
        # self_neg_contrastive_similarity_matrix size： [batch_size, batch_size]

        # mask diagonal
        mask = torch.eye(batch_emb_b.size(0), batch_emb_b.size(0)).bool().to(self.device)
        # tensor mask with diagonal all True others all False
        self_neg_similarity_matrix.masked_fill_(mask, 0)
        contrastive_neg_similarity_matrix.masked_fill_(mask, 0)
        # concatenates tau_1(v) * tau_2(v) with 0-diagonal and tau_1(v) * tau2(neg) with 0-diagonal in row
        # we mask 2 diagonal out because we don't want to get the similarity of an embedding with itself
        neg_contrastive_similarity_matrix = \
            torch.cat([self_neg_similarity_matrix, contrastive_neg_similarity_matrix], -1)
        # sum the matrix up by row
        neg_contrastive_similarity = torch.sum(torch.exp(neg_contrastive_similarity_matrix) / self.args.T_groc, 1)

        loss_vec = -torch.log(contrastive_similarity / (contrastive_similarity + neg_contrastive_similarity))

        return loss_vec

    def groc_loss(self, trn_model, modified_adj_a, modified_adj_b, users, items, mask_1=None, mask_2=None):
        loss_vec_a = self.groc_loss_vec(trn_model, modified_adj_a, modified_adj_b, users, items, mask_1, mask_2)
        loss_vec_b = self.groc_loss_vec(trn_model, modified_adj_b, modified_adj_a, users, items, mask_1, mask_2)

        return torch.sum(torch.add(loss_vec_a, loss_vec_b)) / (2 * loss_vec_a.size(0))

    def get_modified_adj_for_insert(self, batch_nodes, adj_with_2_hops):
        """
        reset flag is a flag that indicate the adj will insert edges(flag==False, do sum) or set the adj back to original adj
        """
        # use one hot embedding get corresponding adj_vectors and get the unique vector that merge all interaction info of these nodes, namely add edges
        i = torch.stack((batch_nodes, batch_nodes))
        v = torch.ones(i.shape[1])
        batch_nodes_in_matrix = torch.sparse_coo_tensor(i, v, adj_with_2_hops.shape).to(self.device)

        # make sure there are no connections between users and users / items and items

        adj_with_2_hops[:self.num_users, :self.num_users] = 0.
        adj_with_2_hops[self.num_users:, self.num_users:] = 0.
        # after modification the adj_with_2_hops is still dense(density > 50%)

        where_to_insert = (torch.sparse.mm(batch_nodes_in_matrix, adj_with_2_hops) -
                           torch.sparse.mm(batch_nodes_in_matrix, self.ori_adj)).to(self.device)

        num_insert = where_to_insert.sum()
        assert num_insert != 0, "you fucked up dude. Check where you build your new Adj matrix"
        adj_with_insert = self.ori_adj + where_to_insert / num_insert

        del where_to_insert
        gc.collect()

        return adj_with_insert

    def get_modified_adj_with_insert_and_remove_by_gradient(self, remove_prob, insert_prob, batch_all_node,
                                                            edge_gradient):
        adj_insert_remove = self.ori_adj.to(self.device)

        tril_adj_index = torch.tril_indices(row=len(adj_insert_remove) - 1, col=len(adj_insert_remove) - 1, offset=0)
        tril_adj_index = tril_adj_index.to(self.device)
        tril_adj_index_0 = tril_adj_index[0][tril_adj_index[0] != tril_adj_index[1]]
        tril_adj_index_1 = tril_adj_index[1][tril_adj_index[0] != tril_adj_index[1]]

        k_remove = int(remove_prob * adj_insert_remove[batch_all_node].sum())
        edge_gradient = (edge_gradient * adj_insert_remove)[tril_adj_index_0, tril_adj_index_1]
        _, indices_rm = torch.topk(edge_gradient, k_remove, largest=False)

        low_tril_matrix = adj_insert_remove[tril_adj_index_0, tril_adj_index_1]
        up_tril_matrix = adj_insert_remove[tril_adj_index_1, tril_adj_index_0]
        low_tril_matrix[indices_rm] = 0.
        up_tril_matrix[indices_rm] = 0.

        k_insert = int(insert_prob * len(batch_all_node) * (len(batch_all_node) - 1) / 2)
        _, indices_ir = torch.topk(edge_gradient, k_insert)
        low_tril_matrix[indices_ir] = 1.
        up_tril_matrix[indices_ir] = 1.

        adj_insert_remove[tril_adj_index_0, tril_adj_index_1] = low_tril_matrix
        adj_insert_remove[tril_adj_index_1, tril_adj_index_0] = up_tril_matrix

        del tril_adj_index
        del tril_adj_index_0
        del tril_adj_index_1

        del low_tril_matrix
        del up_tril_matrix

        gc.collect()

        return adj_insert_remove

    def contruct_adj_after_n_hops(self):
        # for _ in range(1):
        # only consider 2-hop neighbour
        # reason: if onlu 1-hop considered, only add interaction between user-user and item-item.
        # negative: super dense tensor
        adj_after_2_hops = self.ori_adj
        for _ in range(2):
            adj_after_2_hops = ((torch.mm(adj_after_2_hops, adj_after_2_hops) + adj_after_2_hops) > 0.).float()

        gc.collect()

        return adj_after_2_hops

    def attack_adjs(self, adj_a, adj_b, perturbations, users, posItems, negItems):
        modified_adj_a = attack_model(self.ori_model, adj_a, perturbations, self.args.path_modified_adj,
                                      self.args.modified_adj_name_with_rdm_ptb_a, self.args.modified_adj_id,
                                      users, posItems, negItems, self.ori_model.num_users, self.device)

        modified_adj_b = attack_model(self.ori_model, adj_b, perturbations, self.args.path_modified_adj,
                                      self.args.modified_adj_name_with_rdm_ptb_b, self.args.modified_adj_id,
                                      users, posItems, negItems, self.ori_model.num_users, self.device)

        try:
            print("modified adjacency matrix are not same:", (modified_adj_a == modified_adj_b).all())
        except AttributeError:
            print("2 modified adjacency matrix are same. Check your perturbation value")

        return modified_adj_a, modified_adj_b

    def groc_train(self):
        self.ori_model.train()
        embedding_param = []
        adj_param = []
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.ori_model.lr, weight_decay=self.ori_model.weight_decay)

        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, self.num_users + self.num_items, self.args.warmup_steps,
                                       self.args.groc_batch_size,
                                       self.args.groc_epochs)

        all_node_index = torch.arange(0, self.num_users + self.num_items, 1).to(self.device)
        all_node_index = utils.shuffle(all_node_index)

        total_batch = len(all_node_index) // self.args.groc_batch_size + 1

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()
            aver_loss = 0.
            for (batch_i, (batch_all_node)) in \
                    enumerate(utils.minibatch(all_node_index, batch_size=self.args.groc_batch_size)):
                user_filter = (batch_all_node < self.num_users).to(self.device)
                batch_users = torch.masked_select(batch_all_node, user_filter).to(self.device)
                batch_items = torch.sub(torch.masked_select(batch_all_node, ~user_filter), self.num_users).to(
                    self.device)
                adj_with_insert = self.get_modified_adj_for_insert(batch_all_node)  # 2 views are same
                mask_1 = (torch.FloatTensor(self.ori_model.latent_dim).uniform_() < self.args.mask_prob_1).to(
                    self.device)
                mask_2 = (torch.FloatTensor(self.ori_model.latent_dim).uniform_() < self.args.mask_prob_2).to(
                    self.device)

                loss_for_grad = self.groc_loss(self.ori_model, adj_with_insert, adj_with_insert, batch_users,
                                               batch_items, mask_1, mask_2)

                # remove index of diagonal

                edge_gradient = torch.autograd.grad(loss_for_grad, self.ori_model.adj, retain_graph=True)[0]

                adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_1,
                                                                                               self.args.remove_prob_1,
                                                                                               batch_all_node,
                                                                                               edge_gradient)
                adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_2,
                                                                                               self.args.remove_prob_2,
                                                                                               batch_all_node,
                                                                                               edge_gradient)

                loss = self.groc_loss(self.ori_model, adj_insert_remove_1, adj_insert_remove_2, batch_users,
                                      batch_items,
                                      mask_1, mask_2)
                loss.backward()
                optimizer.step()
                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()

            aver_loss = aver_loss / total_batch

            if i % 10 == 0:
                print("GROC Loss: ", aver_loss)

    def bpr_with_dcl(self, data_len_, modified_adj_a, modified_adj_b, users, posItems, negItems):
        self.ori_model.train()
        optimizer = optim.Adam(self.ori_model.parameters(), lr=self.ori_model.lr,
                               weight_decay=self.ori_model.weight_decay)
        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.batch_size,
                                       self.args.groc_epochs)

        total_batch = len(users) // self.args.batch_size + 1

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()

            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss = 0.
            aver_bpr_loss = 0.
            aver_dcl_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=10)):
                self.ori_adj = utils.normalize_adj_tensor(self.ori_adj, sparse=True)
                modified_adj_a = utils.normalize_adj_tensor(modified_adj_a, sparse=True)
                modified_adj_b = utils.normalize_adj_tensor(modified_adj_b, sparse=True)

                gc.collect()

                bpr_loss, reg_loss = self.ori_model.bpr_loss(self.ori_adj, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.ori_model.weight_decay
                dcl_loss = self.groc_loss(self.ori_model, modified_adj_a, modified_adj_b, batch_users, batch_pos)
                loss = self.args.loss_weight_bpr * bpr_loss + reg_loss + (1 - self.args.loss_weight_bpr) * dcl_loss

                loss.backward()
                optimizer.step()
                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()
                aver_bpr_loss += bpr_loss.cpu().item()
                aver_dcl_loss += dcl_loss.cpu().item()

            aver_loss = aver_loss / total_batch
            aver_bpr_loss = aver_bpr_loss / total_batch
            aver_dcl_loss = aver_dcl_loss / total_batch

            if i % 10 == 0:
                print("GROC Loss: ", aver_loss)
                print("BPR Loss: ", aver_bpr_loss)
                print("DCL Loss: ", aver_dcl_loss)

    def groc_train_with_bpr(self, data_len_, users, posItems, negItems):
        self.ori_model.train()
        embedding_param = []
        adj_param = []
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.ori_model.lr, weight_decay=self.ori_model.weight_decay)

        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.groc_batch_size,
                                       self.args.groc_epochs)

        total_batch = len(users) // self.args.batch_size + 1
        ori_adj_sparse = utils.normalize_adj_tensor(self.ori_adj, sparse=True)

        adj_with_2_hops = self.contruct_adj_after_n_hops()

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()

            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss = 0.
            aver_bpr_loss = 0.
            aver_groc_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.batch_size)):

                batch_items = utils.shuffle(torch.cat((batch_pos, batch_neg))).to(self.device)
                batch_all_node = torch.cat((batch_users, batch_items + self.num_users)).unique(sorted=False) \
                    .to(self.device)

                batch_all_node = batch_all_node[:10]
                adj_with_insert = self.get_modified_adj_for_insert(batch_all_node, adj_with_2_hops)  # 2 views are same

                mask_1 = (torch.FloatTensor(self.ori_model.latent_dim).uniform_() < self.args.mask_prob_1) \
                    .to(self.device)
                mask_2 = (torch.FloatTensor(self.ori_model.latent_dim).uniform_() < self.args.mask_prob_2) \
                    .to(self.device)

                # batch_users_groc = batch_all_node[batch_all_node < self.num_users]
                # batch_items = batch_all_node[batch_all_node >= self.num_users] - self.num_users

                adj_for_loss_gradient = utils.normalize_adj_tensor(adj_with_insert, sparse=True)

                loss_for_grad = self.ori_gcl_computing(self.ori_model, adj_for_loss_gradient, adj_for_loss_gradient,
                                                       batch_users, batch_pos, mask_1, mask_2, query_groc=True)

                # remove index of diagonal

                edge_gradient = torch.autograd.grad(loss_for_grad, self.ori_model.adj, retain_graph=True)[0].to_dense()

                del adj_with_insert
                del adj_for_loss_gradient
                gc.collect()

                adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_1,
                                                                                               self.args.remove_prob_1,
                                                                                               batch_all_node,
                                                                                               edge_gradient)
                adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_2,
                                                                                               self.args.remove_prob_2,
                                                                                               batch_all_node,
                                                                                               edge_gradient)

                groc_loss = self.ori_gcl_computing(self.ori_model,
                                                   utils.normalize_adj_tensor(adj_insert_remove_1, sparse=True),
                                                   utils.normalize_adj_tensor(adj_insert_remove_2, sparse=True),
                                                   batch_users, batch_pos, mask_1, mask_2)

                del adj_insert_remove_1
                del adj_insert_remove_2

                bpr_loss, reg_loss = self.ori_model.bpr_loss(ori_adj_sparse, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.ori_model.weight_decay

                loss = self.args.loss_weight_bpr * bpr_loss + reg_loss + (1 - self.args.loss_weight_bpr) * groc_loss
                loss.backward()
                optimizer.step()

                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()
                aver_bpr_loss += bpr_loss.cpu().item()
                aver_groc_loss += groc_loss.cpu().item()

            aver_loss = aver_loss / total_batch
            aver_bpr_loss = aver_bpr_loss / total_batch
            aver_dcl_loss = aver_groc_loss / total_batch

            if i % 10 == 0:
                print("GROC Loss: ", aver_loss)
                print("BPR Loss: ", aver_bpr_loss)
                print("DCL Loss: ", aver_dcl_loss)

    def ori_gcl_computing(self, trn_model, gra1, gra2, users, poss, mask_1=None, mask_2=None, query_groc=None):
        (user_emb, _, _, _) = trn_model.getEmbedding(self.ori_adj, users.long(), poss.long())

        (users_emb_perturb_1, _, _, _) = trn_model.getEmbedding(gra1, users.long(), poss.long(), query_groc=query_groc)
        if mask_1 is not None:
            users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1).masked_fill_(mask_1, 0.)
        else:
            users_emb_perturb_1 = nn.functional.normalize(users_emb_perturb_1, dim=1)
        (users_emb_perturb_2, _, _, _) = trn_model.getEmbedding(gra2, users.long(), poss.long(), query_groc=query_groc)
        if mask_2 is not None:
            users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1).masked_fill_(mask_2, 0.)
        else:
            users_emb_perturb_2 = nn.functional.normalize(users_emb_perturb_2, dim=1)
        users_dot_12 = torch.bmm(users_emb_perturb_1.unsqueeze(1), users_emb_perturb_2.unsqueeze(2)).squeeze(2)
        users_dot_12 /= self.args.T_groc
        fenzi_12 = torch.exp(users_dot_12).sum(1)

        neg_emb_users_12 = users_emb_perturb_2.unsqueeze(0).repeat(user_emb.size(0), 1, 1)
        neg_dot_12 = torch.bmm(neg_emb_users_12, users_emb_perturb_1.unsqueeze(2)).squeeze(2)
        neg_dot_12 /= self.args.T_groc
        neg_dot_12 = torch.exp(neg_dot_12).sum(1)

        mask_11 = self.get_negative_mask_perturb(users_emb_perturb_1.size(0)).to(self.device)
        neg_dot_11 = torch.exp(torch.mm(users_emb_perturb_1, users_emb_perturb_1.t()) / self.args.T_groc)
        neg_dot_11 = neg_dot_11.masked_select(mask_11).view(users_emb_perturb_1.size(0), -1).sum(1)
        loss_perturb_11 = (-torch.log(fenzi_12 / (neg_dot_11 + neg_dot_12))).mean()

        users_dot_21 = torch.bmm(users_emb_perturb_2.unsqueeze(1), users_emb_perturb_1.unsqueeze(2)).squeeze(2)
        users_dot_21 /= self.args.T_groc
        fenzi_21 = torch.exp(users_dot_21).sum(1)

        neg_emb_users_21 = users_emb_perturb_1.unsqueeze(0).repeat(user_emb.size(0), 1, 1)
        neg_dot_21 = torch.bmm(neg_emb_users_21, users_emb_perturb_2.unsqueeze(2)).squeeze(2)
        neg_dot_21 /= self.args.T_groc
        neg_dot_21 = torch.exp(neg_dot_21).sum(1)

        mask_22 = self.get_negative_mask_perturb(users_emb_perturb_2.size(0)).to(self.device)
        neg_dot_22 = torch.exp(torch.mm(users_emb_perturb_2, users_emb_perturb_2.t()) / self.args.T_groc)
        neg_dot_22 = neg_dot_22.masked_select(mask_22).view(users_emb_perturb_2.size(0), -1).sum(1)
        loss_perturb_22 = (-torch.log(fenzi_21 / (neg_dot_22 + neg_dot_21))).mean()

        loss_perturb = loss_perturb_11 + loss_perturb_22

        return loss_perturb

    @staticmethod
    def get_negative_mask_perturb(batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0

        return negative_mask

    def ori_gcl_train_with_bpr(self, gra1, gra2, data_len_, users, posItems, negItems):
        self.ori_adj = utils.normalize_adj_tensor(self.ori_adj, sparse=True)
        gra1 = utils.normalize_adj_tensor(gra1, sparse=True)
        gra2 = utils.normalize_adj_tensor(gra2, sparse=True)

        gc.collect()

        self.ori_model.train()
        embedding_param = []
        adj_param = []
        for n, p in self.ori_model.named_parameters():
            if n.__contains__('embedding'):
                embedding_param.append(p)
            else:
                adj_param.append(p)
        optimizer = optim.Adam([
            {'params': embedding_param},
            {'params': adj_param, 'lr': 0}
        ], lr=self.ori_model.lr, weight_decay=self.ori_model.weight_decay)

        if self.args.use_scheduler:
            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.groc_batch_size,
                                       self.args.groc_epochs)

        total_batch = len(users) // self.args.batch_size + 1

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()

            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)

            aver_loss = 0.
            aver_bpr_loss = 0.
            aver_groc_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.batch_size)):

                gcl = self.ori_gcl_computing(self.ori_model, gra1, gra2, batch_users, batch_pos)

                bpr_loss, reg_loss = self.ori_model.bpr_loss(self.ori_adj, batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.ori_model.weight_decay

                loss = self.args.loss_weight_bpr * bpr_loss + reg_loss + (1 - self.args.loss_weight_bpr) * gcl
                loss.backward()
                optimizer.step()

                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()
                aver_bpr_loss += bpr_loss.cpu().item()
                aver_groc_loss += gcl.cpu().item()

            aver_loss = aver_loss / total_batch
            aver_bpr_loss = aver_bpr_loss / total_batch
            aver_dcl_loss = aver_groc_loss / total_batch

            if i % 10 == 0:
                print("GROC Loss: ", aver_loss)
                print("BPR Loss: ", aver_bpr_loss)
                print("DCL Loss: ", aver_dcl_loss)

    def fit(self):
        pass
