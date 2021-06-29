import torch
from torch import nn,optim
import numpy as np
from register import dataset
import utils

class LightGCN(nn.Module):
    def __init__(self,device=None):
        super(LightGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device=device
        self.lr=0.001
        self.weight_decay=1e-4
        self.n_layers=3
        self.num_users=dataset.n_user
        self.num_items=dataset.m_item
        self.latent_dim=64
        self.f = nn.Sigmoid()

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

    def fit(self, adj, users, posItems, negItems):
        if type(adj) is not torch.Tensor:
            adj_norm=utils.normalize_adj_tensor(adj)
            adj=utils.to_tensor(adj_norm,device=self.device)
        else:
            adj_norm=utils.normalize_adj_tensor(adj)
            adj=adj_norm.to(self.device)
        #self.adj=adj

        self._train_without_val(adj, users, posItems, negItems)

    def getUsersRating(self,adj,users):
        all_users,all_items=self.computer(adj)
        users_emb=all_users[users.long()]
        items_emb=all_items
        rating=self.f(torch.matmul(users_emb,items_emb.t()))
        return rating

    def computer(self,adj):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]

        g_droped = adj

        for layer in range(self.n_layers):
            all_emb = torch.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, adj, users, items):
        # compute embedding
        all_users, all_items = self.computer(adj)
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


    def getEmbedding(self, adj, users, pos_items, neg_items):
        all_users, all_items = self.computer(adj)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def bpr_loss(self,adj,users,pos,neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(adj, users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def _train_without_val(self,adj, users, posItems, negItems):
            self.train()
            optimizer=optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            for i in range(100):
                optimizer.zero_grad()
                users = users.to(self.device)
                posItems = posItems.to(self.device)
                negItems = negItems.to(self.device)
                users, posItems, negItems = utils.shuffle(users, posItems, negItems)
                total_batch = len(users) // 2048 + 1
                aver_loss = 0.
                for (batch_i,
                     (batch_users,
                      batch_pos,
                      batch_neg)) in enumerate(utils.minibatch(users,
                                                               posItems,
                                                               negItems,
                                                               batch_size=2048)):
                    loss, reg_loss = self.bpr_loss(adj, batch_users, batch_pos, batch_neg)
                    reg_loss = reg_loss * self.weight_decay
                    loss = loss + reg_loss
                    
                    
                    loss.backward()
                    optimizer.step()
                    
                    aver_loss+=loss.cpu().item()
                aver_loss = aver_loss / total_batch
                if i%10==0:
                    print(aver_loss)

            self.eval()
            
    
            #users = users.to(self.device)
            #posItems = posItems.to(self.device)
            #negItems = negItems.to(self.device)
            #users, posItems, negItems = utils.shuffle(users, posItems, negItems)
            output=self.forward(adj, users, posItems)
            self.output=output