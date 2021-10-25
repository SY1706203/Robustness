import torch
import gc
import numpy as np
from lightgcn import LightGCN
from GraphContrastiveLoss import ori_gcl_computing


class IntegratedGradients:
    def __init__(self, model, args, device, sparse: bool):
        self._is_sparse = sparse
        self.model = model
        self.device = device
        self.args = args

        if not isinstance(self.model, LightGCN):
            raise TypeError(
                "Currently the IntegratedGradient calculation only support for model LightGCN."
            )

    def get_integrated_gradient(self, adj, ori_model, ori_adj, batch_users, batch_items, mask_1, mask_2,
                                non_exist_edge=False, adj_baseline=None, steps=5):

        if adj_baseline is None:
            if non_exist_edge:
                adj_baseline = torch.ones(adj.shape).to(self.device)
            else:

                i = torch.tensor([[], []]).to(self.device)
                v = torch.tensor([]).to(self.device)
                adj_baseline = torch.sparse_coo_tensor(i, v, ori_adj.shape).to(self.device)

        adj_diff = adj - adj_baseline

        total_gradients = torch.zeros_like(adj).to(self.device)

        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            adj_step = adj_baseline + alpha * adj_diff

            loss_for_grad = ori_gcl_computing(ori_adj, ori_model, adj_step, adj_step, batch_users, batch_items,
                                              self.args, self.device, mask_1=mask_1, mask_2=mask_2, query_groc=True)

            # assert ori_model.adj == adj_step, "Sorry it's fucked up bro. Report the bug to the author pls, Thank you very much"
            grads = torch.autograd.grad(loss_for_grad, ori_model.adj, retain_graph=True)[0]

            del adj_step

            total_gradients += grads

            del grads

        gc.collect()

        if self._is_sparse:
            total_gradients = torch.sparse.mm(total_gradients, adj_diff) / steps
        else:
            total_gradients = torch.mm(total_gradients, adj_diff) / steps

        return total_gradients

