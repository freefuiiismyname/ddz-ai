# -*- coding: utf-8 -*-
import random
import tensorflow as tf


def sample(combos, combo_l_mask, combo_a_mask, sample_num=100):
    """
    动作空间太大，则将采样
    """
    if len(combos) <= sample_num:
        return combos, combo_l_mask, combo_a_mask
    else:
        idxes = random.sample([i for i in range(len(combos))], sample_num)
        asid, lmsk, amsk = [], [], []
        for idx in idxes:
            asid.append(combos[idx])
            lmsk.append(combo_l_mask[idx])
            amsk.append(combo_a_mask[idx])
        return asid, lmsk, amsk


def convert_h(h_vec, h_pid):
    """
    对入参填充和定义mask
    """
    max_len = max([len(i) for i in h_pid])
    h_mask = [[1 for _ in i] for i in h_pid]
    h_vec = pad_2d(h_vec, max_len, pad_id=[0, 0, 0, 0, 0])
    h_mask = pad_2d(h_mask, max_len, pad_id=0)
    h_pid = pad_2d(h_pid, max_len, pad_id=0)

    return h_vec, h_pid, h_mask


def pad_2d(tensor, max_len, pad_id):
    """
    填充成矩阵
    """
    res = []
    for i in tensor:
        j = []
        for k in i:
            j.append(k)
        while len(j) < max_len:
            j.append(pad_id)
        res.append(j)
    return res


def convert_alad(act_space_ids, label_masks, attn_masks, dynamic_corpus):
    """
    对入参填充
    """
    dynamic_corpus = pad_3d(dynamic_corpus, pad_id=0)
    act_space_ids = pad_3d(act_space_ids, pad_id=0)
    label_masks = pad_3d(label_masks, pad_id=1)
    attn_masks = pad_3d(attn_masks, pad_id=0)
    return act_space_ids, label_masks, attn_masks, dynamic_corpus


def convert_aad(act_space_ids, attn_masks, dynamic_corpus):
    """
    对入参进行采样和填充
    """
    dynamic_corpus = pad_3d(dynamic_corpus, pad_id=0)
    act_space_ids = pad_3d(act_space_ids, pad_id=0)
    attn_masks = pad_3d(attn_masks, pad_id=0)
    return act_space_ids, attn_masks, dynamic_corpus


def pad_3d(input_list, pad_id):
    """
    填充成矩阵
    """
    max_1d = max([len(i) for i in input_list])
    max_2d = max([max([len(j) for j in i]) for i in input_list])

    tensor = []
    for i in input_list:
        d1 = []
        for j in i:
            d2 = []
            for k in j:
                d2.append(k)
            while len(d2) < max_2d:
                d2.append(pad_id)
            d1.append(d2)
        while len(d1) < max_1d:
            d2 = []
            for _ in range(max_2d):
                d2.append(pad_id)
            d1.append(d2)
        tensor.append(d1)

    return tensor


def clip_train_op(loss, params, optimizer):
    """
    梯度裁剪训练
    """
    grads = tf.gradients(loss, params)
    real_grads = []
    real_vars = []
    for d_n, dn_v in zip(grads, params):
        if d_n == None:
            print('None:::', dn_v.name)
            continue
        real_grads.append(d_n)
        real_vars.append(dn_v)
    grads = real_grads
    params = real_vars
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=2.0)
    train_op = optimizer.apply_gradients(zip(grads, params))

    return train_op

