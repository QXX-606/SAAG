import os
print(os.getcwd())
import torch
import torch.optim as optim
from GCN.NGCF import NGCF
from GCN.utility.helper import *
from GCN.utility.batch_test import *
import argparse
from ast import parse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
from time import time
import Diff_att.models.gaussian_diffusion as gd
from Diff_att.models.DNN import DNN
from Diff_att.models.multivae import MultiVAE
import Diff_att.evaluate_utils
import Diff_att.data_utils as data_utils
from copy import deepcopy

if __name__ == '__main__':
    args.device = torch.device('cuda:' + str(args.gpu_id))
    # 读取属性数据
    att_train_path = args.data_path +args.dataset+ '/att.npy'
    att_train_data = np.load(att_train_path, allow_pickle=True)
    n_vae = 64
    rating_data = args.data_path +args.dataset+ '/rating_matrix.pt'
    rating_matrix = torch.load(rating_data)
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(att_train_data))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=4)
    train_loader = torch.FloatTensor(att_train_data)
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)
    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
            args.noise_scale, args.noise_min, args.noise_max, args.steps, args.device).to(args.device)
    out_dims = eval(args.dims) + [n_vae]
    in_dims = out_dims[::-1]
    DNN_model = DNN(in_dims, out_dims, args.emb_size,time_type="cat", norm=args.norm).to(args.device)
    gvae= MultiVAE(rating_matrix,args.device).to(args.device)
    para_sum = list(DNN_model.parameters())
    para_sum += list(gvae.parameters())
    print("models ready.")
    param_num = 0
    mlp_num = sum([param.nelement() for param in DNN_model.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
    param_num = mlp_num + diff_num
    print("Number of all parameters:", param_num)
    train_path = args.data_path + 'att.npy'
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    t0 = time()
    cur_best_pre_0, stopping_step = 0, 0
    para_sum += list(model.parameters())
    #optimizer = optim.AdamW(para_sum, lr=args.lr)
    optimizer = optim.Adam(para_sum, lr=args.lr)
    #optimizer2 = optim.Adam(para_sum, lr=args.lr)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lamda1,lamda2 = 0.01,0.001
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        DNN_model.train()
        gvae.train()
        loss, mf_loss, emb_loss,vae_loss,diff_loss,gnn_loss = 0., 0., 0., 0., 0.,0.
        n_batch = data_generator.n_train // args.batch_size + 1
        loss_vae,struct_emb = gvae.calculate_loss(rating_matrix)
        print(struct_emb.shape)
        losses,q = diffusion.training_losses(DNN_model, train_loader, struct_emb, args.reweight)
        model.updata_att_emb(q)
        diff_loss = losses["loss"].mean()
        loss_batch_total = 0
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                            pos_items,
                                                                            neg_items,-1,
                                                                            drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                                pos_i_g_embeddings,
                                                                                neg_i_g_embeddings)
            loss_batch_total += batch_loss
        loss_total = diff_loss*lamda1 + loss_vae*lamda2 + loss_batch_total
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        loss += loss_total
        gnn_loss += batch_loss
        mf_loss += batch_mf_loss
        emb_loss += batch_emb_loss
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f ]' % (
                    epoch+1, time() - t1, loss,loss_batch_total,diff_loss*lamda1,loss_vae*lamda2)
                print(perf_str)
            continue
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)
        t3 = time()
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)
        if should_stop == True:
            break
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)
    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

