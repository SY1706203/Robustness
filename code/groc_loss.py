import torch
import torch.nn as nn
import torch.optim as optim
from utils import scheduler_groc
import utils
import torch.nn.functional as F
from utils_attack import attack_model


class GROC_loss:
    def __init__(self, ori_model, args, users, posItems, negItems):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ori_model = ori_model
        self.args = args
        self.users = users
        self.posItems = posItems
        self.negItems = negItems
        self.modified_adj_a = None
        self.modified_adj_b = None
        self.total_batch = None

    @staticmethod
    def get_embed_groc(trn_model, modified_adj, users_, poss):
        (users_emb, pos_emb, _, _) = trn_model.getEmbedding(modified_adj, users_.long(), poss.long())

        users_emb = nn.functional.normalize(users_emb, dim=1)
        pos_emb = nn.functional.normalize(pos_emb, dim=1)

        return torch.cat([users_emb, pos_emb])

    def groc_loss_vec(self, trn_model, modified_adj_a, modified_adj_b, users_, poss):
        batch_emb_a = self.get_embed_groc(trn_model, modified_adj_a, users_, poss)
        batch_emb_b = self.get_embed_groc(trn_model, modified_adj_b, users_, poss)

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

    def groc_loss(self, trn_model, users_, poss):
        loss_vec_a = self.groc_loss_vec(trn_model, self.modified_adj_a, self.modified_adj_b, users_, poss)
        loss_vec_b = self.groc_loss_vec(trn_model, self.modified_adj_b, self.modified_adj_a, users_, poss)

        return torch.sum(torch.add(loss_vec_a, loss_vec_b)) / (2 * loss_vec_a.size(0))

    def attack_adjs(self, adj_a, adj_b, perturbations, users):
        self.modified_adj_a = attack_model(self.ori_model, adj_a, perturbations, self.args.path_modified_adj,
                                           self.args.modified_adj_name_with_rdm_ptb, self.args.modified_adj_id,
                                           self.users,
                                           self.posItems, self.negItems, self.ori_model.num_users, self.device)

        self.modified_adj_b = attack_model(self.ori_model, adj_b, perturbations, self.args.path_modified_adj,
                                           ['a_02_w_r_b', 'a_04_w_r_b', 'a_06_w_r_b', 'a_08_w_r_b', 'a_1_w_r_b',
                                            'a_12_w_r_b', 'a_14_w_r_b', 'a_16_w_r_b', 'a_18_w_r_b', 'a_2_w_r_b'],
                                           self.args.modified_adj_id, self.users, self.posItems, self.negItems,
                                           self.ori_model.num_users, self.device)

        try:
            print("modified adjacency matrix are not same:", (self.modified_adj_a == self.modified_adj_b).all())
        except AttributeError:
            print("2 modified adjacency matrix are same. Check your perturbation value")

        self.total_batch = len(users) // self.args.batch_size + 1

    def groc_train(self, data_len_, adj_a, adj_b, modified_adj, model, perturbations, users):
        if self.modified_adj_a is None or self.modified_adj_b is None:
            self.attack_adjs(adj_a, adj_b, perturbations, users)

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=model.lr,
                               weight_decay=model.weight_decay)

        if self.args.use_scheduler:

            scheduler = scheduler_groc(optimizer, data_len_, self.args.warmup_steps, self.args.batch_size,
                                       self.args.groc_epochs)

        for i in range(self.args.groc_epochs):
            optimizer.zero_grad()
            users_ = users.to(self.device)
            posItems_ = self.posItems.to(self.device)
            negItems_ = self.negItems.to(self.device)
            users_, posItems_, negItems_ = utils.shuffle(users_, posItems_, negItems_)
            aver_loss = 0.
            aver_bpr_loss = 0.
            aver_dcl_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) \
                    in enumerate(utils.minibatch(users_, posItems_, negItems_, batch_size=self.args.batch_size)):
                if self.args.train_groc_casade:
                    loss = self.groc_loss(model, batch_users, batch_pos)
                else:
                    bpr_loss, reg_loss = model.bpr_loss(modified_adj, batch_users, batch_pos, batch_neg)
                    reg_loss = reg_loss * model.weight_decay
                    dcl_loss = self.groc_loss(model, batch_users, batch_pos)
                    loss = self.args.loss_weight_bpr * bpr_loss + reg_loss + (1 - self.args.loss_weight_bpr) * dcl_loss

                loss.backward()
                optimizer.step()
                if self.args.use_scheduler:
                    scheduler.step()

                aver_loss += loss.cpu().item()
                if not self.args.train_groc_casade:
                    aver_bpr_loss += bpr_loss.cpu().item()
                    aver_dcl_loss += dcl_loss.cpu().item()

            aver_loss = aver_loss / self.total_batch
            aver_bpr_loss = aver_bpr_loss / self.total_batch
            aver_dcl_loss = aver_dcl_loss / self.total_batch
            if i % 10 == 0:
                print("GROC Loss: ", aver_loss)
                if not self.args.train_groc_casade:
                    print("BPR Loss: ", aver_bpr_loss)
                    print("DCL Loss: ", aver_dcl_loss)
