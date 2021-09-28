import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import scheduler_groc
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

        adj_norm = utils.normalize_adj_tensor(modified_adj)
        modified_adj = adj_norm.to(self.device)

        (users_emb, item_emb, _, _) = trn_model.getEmbedding(modified_adj, users.long(), items.long(), query_groc=True)

        users_emb = nn.functional.normalize(users_emb, dim=1).masked_fill_(mask, 0)
        item_emb = nn.functional.normalize(item_emb, dim=1).masked_fill_(mask, 0)

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

    def groc_loss(self, trn_model, modified_adj_a, modified_adj_b, users, items, mask_1, mask_2):
        loss_vec_a = self.groc_loss_vec(trn_model, modified_adj_a, modified_adj_b, users, items, mask_1, mask_2)
        loss_vec_b = self.groc_loss_vec(trn_model, modified_adj_b, modified_adj_a, users, items, mask_1, mask_2)

        return torch.sum(torch.add(loss_vec_a, loss_vec_b)) / (2 * loss_vec_a.size(0))

    def get_modified_adj_for_insert(self, batch_nodes):
        """
        reset flag is a flag that indicate the adj will insert edges(flag==False, do sum) or set the adj back to original adj
        """
        nodes_num = len(batch_nodes)
        num_edges_insertion = int(nodes_num * (nodes_num - 1) / 2)
        index_array = np.stack((batch_nodes.cpu().numpy().repeat(nodes_num), np.tile(batch_nodes.cpu().numpy(), nodes_num)))
        adj_with_insert = self.ori_adj.clone().to(self.device)
        adj_with_insert[index_array[0], index_array[1]] = adj_with_insert[index_array] + 1 / num_edges_insertion
        adj_with_insert[index_array[1], index_array[0]] = adj_with_insert[index_array] + 1 / num_edges_insertion

        return adj_with_insert.masked_fill_(torch.eye(adj_with_insert.size(0), adj_with_insert.size(0)).bool().to(self.device), 0)

    def get_modified_adj_with_insert_and_remove_by_gradient(self, remove_prob, insert_prob, batch_all_node, edge_gradient):
        adj_insert_remove = self.ori_adj.clone().to(self.device)

        k_remove = int(remove_prob * adj_insert_remove[batch_all_node].sum())
        edge_gradient = edge_gradient * self.ori_adj
        _, indices_rm = torch.topk(edge_gradient, k_remove, largest=False)
        adj_insert_remove[indices_rm[0], indices_rm[1]] = 0
        adj_insert_remove[indices_rm[1], indices_rm[0]] = 0

        k_insert = int(insert_prob * len(batch_all_node) * (len(batch_all_node) - 1) / 2)
        _, indices_ir = torch.topk(edge_gradient, k_insert)
        adj_insert_remove[indices_ir[0], indices_ir[1]] = 1
        adj_insert_remove[indices_ir[1], indices_ir[0]] = 1

        return adj_insert_remove

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
            scheduler = scheduler_groc(optimizer, self.num_users+self.num_items, self.args.warmup_steps, self.args.groc_batch_size,
                                       self.args.groc_epochs)

        all_node_index = torch.arange(0, self.num_users + self.num_items, 1).to(self.device)
        all_node_index = utils.shuffle(all_node_index)

        total_batch = len(all_node_index) // self.args.groc_batch_size + 1

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()
            aver_loss = 0.
            for (batch_i, (batch_all_node)) in enumerate(utils.minibatch(all_node_index, batch_size=self.args.groc_batch_size)):
                user_filter = (batch_all_node < self.num_users).to(self.device)
                batch_users = torch.masked_select(batch_all_node, user_filter).to(self.device)
                batch_items = torch.sub(torch.masked_select(batch_all_node, ~user_filter), self.num_users).to(self.device)
                adj_with_insert = self.get_modified_adj_for_insert(batch_all_node)  # 2 views are same
                mask_1 = (torch.FloatTensor(self.ori_model.latent_dim).uniform_() < self.args.mask_prob_1).to(self.device)
                mask_2 = (torch.FloatTensor(self.ori_model.latent_dim).uniform_() < self.args.mask_prob_2).to(self.device)

                loss_for_grad = self.groc_loss(self.ori_model, adj_with_insert, adj_with_insert, batch_users, batch_items,
                                               mask_1, mask_2)
                edge_gradient = torch.autograd.grad(loss_for_grad, self.ori_model.adj, retain_graph=True)[0]

                adj_insert_remove_1 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_1,
                                                                                               self.args.remove_prob_1,
                                                                                               batch_all_node, edge_gradient)
                adj_insert_remove_2 = self.get_modified_adj_with_insert_and_remove_by_gradient(self.args.insert_prob_2,
                                                                                               self.args.remove_prob_2,
                                                                                               batch_all_node, edge_gradient)

                loss = self.groc_loss(self.ori_model, adj_insert_remove_1, adj_insert_remove_2, batch_users, batch_items,
                                      mask_1, mask_2)
                loss.backward()
                optimizer.step()
                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()

            aver_loss = aver_loss / total_batch

            if i % 10 == 0:
                print("GROC Loss: ", aver_loss)
