# coding:utf-8
# This part is about data processing.
# Divide the data into three parts and store them in a required
# format,(uid,sid)
import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = '/home/dingziyun/vae_cf/RatData/'

raw_data = pd.read_csv(os.path.join(DATA_DIR, '0605_15_20rats.csv'), header=0)
raw_data = raw_data[raw_data['rating2'] > 0.15]

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):

    if min_sc > 0:
        itemcount = get_count(tp, 'video_id')
        tp = tp[tp['video_id'].isin(itemcount.index[itemcount >= min_sc])]

    if min_uc > 0:
        usercount = get_count(tp, 'did')
        tp = tp[tp['did'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'did'), get_count(tp, 'video_id')
    return tp, usercount, itemcount


raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=2)

sparsity = 1. * raw_data.shape[0] / \
    (user_activity.shape[0] * item_popularity.shape[0])

unique_uid = user_activity.index

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]


n_users = unique_uid.size
n_heldout_users = 13000

tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[int((n_users - n_heldout_users * 2) / 10):int((n_users - n_heldout_users) / 10)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['did'].isin(tr_users)]
unique_sid = pd.unique(train_plays['video_id'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

pro_dir = os.path.join(DATA_DIR, 'pro_sg')

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)


def split_train_test_proportion(data, test_prop=0.2,min_uc=1):
    data_grouped_by_user = data.groupby('did')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)
        # print(i,n_items_u)

        if n_items_u >= min_uc: 
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u,
                                 size=int(test_prop * n_items_u),
                                 replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


vad_plays = raw_data.loc[raw_data['did'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['video_id'].isin(unique_sid)]

vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays,test_prop=0.5)

test_plays = raw_data.loc[raw_data['did'].isin(te_users)]
test_plays = test_plays.loc[test_plays['video_id'].isin(unique_sid)]

test_plays_tr, test_plays_te = split_train_test_proportion(test_plays,test_prop=0.5 )


def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['did']))
    sid = list(map(lambda x: show2id[x], tp['video_id']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


train_data = numerize(train_plays)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)
