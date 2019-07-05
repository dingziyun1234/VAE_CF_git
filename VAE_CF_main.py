# coding: utf-8
from VAE_CF_model import MultiVAE
from VAE_CF_train import p_dims, total_anneal_steps, anneal_cap, arch_str
from VAE_CF_train import load_tr_te_data, NDCG_binary_at_k_batch, Recall_at_k_batch

import os
import numpy as np
from scipy import sparse

import tensorflow as tf

DATA_DIR = '/home/dingziyun/vae_cf/RatData/'
pro_dir = os.path.join(DATA_DIR, 'pro_sg')

##################  加载测试数据并计算测试指标  #######################

test_data_tr, test_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'test_tr.csv'),
    os.path.join(pro_dir, 'test_te.csv'))

N_test = test_data_tr.shape[0]
idxlist_test = range(N_test)

batch_size_test = 2000

tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0)
saver, logits_var, _, _, _ = vae.build_graph()
###################   在验证集上加载性能最佳的模型     ##################################
chkpt_dir = os.path.join(DATA_DIR, 'log/VAE_anneal{}K_cap{:1.1E}/{}').format(
    total_anneal_steps / 1000, anneal_cap, arch_str)
# chkpt_dir = '/Users/dingziyun/Downloads/ml-20m/log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
#     total_anneal_steps/1000, anneal_cap, arch_str)
print("chkpt directory: %s" % chkpt_dir)
#####################    测试数据集  ##########################################
n100_list, r20_list, r50_list = [], [], []
# pred_val_all = []
filename = os.path.join(DATA_DIR, 'pred_val_all.txt')
# filename = '/Users/dingziyun/Downloads/ml-20m/pred_val_all.txt'


def text_save(filename, data):
    file = open(filename, 'a')
    for i in range(len(data)):
        tex = list(data[i])
        s = ''
        for x in tex:
            s += str(x) + ' '
        # s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


with tf.Session() as sess:
    saver.restore(sess, '{}/model'.format(chkpt_dir))

    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf  # np.array shape=[2000，20108]

        # text_save(filename, pred_val)

        # pred_val_all.append(pred_val)

        n100_list.append(NDCG_binary_at_k_batch(
            pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
        r20_list.append(Recall_at_k_batch(
            pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        r50_list.append(Recall_at_k_batch(
            pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))


n100_list = np.concatenate(n100_list)
r20_list = np.concatenate(r20_list)
r50_list = np.concatenate(r50_list)


print(
    "Test NDCG@100=%.5f (%.5f)" %
    (np.mean(n100_list),
     np.std(n100_list) /
     np.sqrt(
        len(n100_list))))
print(
    "Test Recall@20=%.5f (%.5f)" %
    (np.mean(r20_list),
     np.std(r20_list) /
     np.sqrt(
        len(r20_list))))
print(
    "Test Recall@50=%.5f (%.5f)" %
    (np.mean(r50_list),
     np.std(r50_list) /
     np.sqrt(
        len(r50_list))))
