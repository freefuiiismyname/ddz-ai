# -*- coding: utf-8 -*-
from __future__ import absolute_import
from env.game import Game
from rl.modeling import RL
from copy import deepcopy
from env.cards import DynamicCorpus
import random

if __name__ == "__main__":

    step = 0
    start_iter = 0
    init_checkpoint = None
    # init_checkpoint = 'data/rl_50000.ckpt'
    num_epochs = 2000001
    dim_states = 52
    rl = RL(dim_states, lr_a=0.0001, lr_c=0.0001, init_checkpoint=init_checkpoint)
    y_loss, f_loss, e_loss = [0], [0, 0], [0]

    # fine-tune
    if start_iter != 0 and not init_checkpoint:
        rl.load_model('rl', start_iter)

    env = Game()
    for episode in range(start_iter, num_epochs):
        env.reset()
        # 体验采样，由评估器分析此局是否值得进行游戏
        y_memory = {}

        cache_memory = {}
        reward_memory = {}
        history_vec = []
        history_pid = []
        e_act_ids, e_dyn_vec, e_label_mask, e_attn_mask = None, None, None, None
        e_action = -1
        while 1:
            pid = env.now_player_id
            # 无人叫地主 or 游戏结束，记录所有存档
            if env.landlord_count == 3 or env.winner >= 0:
                cls = 0
                # 记录对局
                for _ in range(3):
                    state, f_reward, y_reward, act_ids, dyn_vec, _, label_mask, attn_mask = env.observe(pid)
                    reward_memory[pid] = f_reward
                    # 地主赢
                    if y_reward > 1.5:
                        cls = 1
                    if len(act_ids) == 0:
                        act_ids, attn_mask, label_mask = [[0, 0]], [[1, 0]], [[0, 1]]
                    # 存储抢地主样本
                    if pid in y_memory:
                        (_state, _act_ids, _attn_mask, _dyn_vec) = y_memory[pid]
                        if abs(y_reward) > 1.5:
                            rl.store_y(_state, _act_ids, _attn_mask, _dyn_vec, cls, random.randint(0, 1), [0, 1])
                        else:
                            if y_reward > 0:
                                rl.store_y(_state, _act_ids, _attn_mask, _dyn_vec, random.randint(0, 1), 1, [1, 0])
                            else:
                                rl.store_y(_state, _act_ids, _attn_mask, _dyn_vec, random.randint(0, 1), 0, [1, 0])

                    pid = (pid + 1) % 3
                    env.now_player_id = pid
                if e_action != -1:
                    rl.store_e(e_act_ids, e_dyn_vec, e_label_mask, e_attn_mask, cls)
                for k, v in cache_memory.items():
                    reward = reward_memory[k]
                    for discount_time, mem in enumerate(v[::-1]):
                        (_state, _act_ids, _label_mask, _attn_mask, _dyn_vec, _action, _h_vec, _h_pid, old_p) = mem
                        rl.store_f(_state, _act_ids, _label_mask, _attn_mask, _dyn_vec, _action, _h_vec,
                                   _h_pid, reward, old_p)
                        reward *= 0.95
                step += 1
                break

            state, f_reward, _, act_ids, dyn_vec, id2combo, label_mask, attn_mask = env.observe(pid)

            # 叫地主环节
            if env.landlord_cards:
                action = rl.y_act(state, act_ids, attn_mask, dyn_vec, is_training=True)

                # 更新记忆器
                y_memory[pid] = (state, act_ids, attn_mask, dyn_vec)
                env.step(action)
                if action > 0:
                    history_vec.append([-1, -1, -1, -1, -1])
                    history_pid.append(pid)

                    e_act_ids, e_dyn_vec, e_label_mask, e_attn_mask = env.observe_entirety()
                    # 判断是否值得游戏
                    e_action, next_epi = rl.e_act(e_act_ids, e_dyn_vec, e_label_mask, e_attn_mask)
                    if next_epi:
                        break
            # 斗地主环节
            else:
                tmp_act_ids = [[j for j, lmm in zip(i, lm) if j > 0 and lmm == 0] for i, lm in zip(act_ids, label_mask)]
                tmp_l_m = [[lmm for j, lmm in zip(i, lm) if j > 0 and lmm == 0] for i, lm in zip(act_ids, label_mask)]

                # 出不了牌
                if max([len(i) for i in tmp_l_m] + [0]) == 0:
                    env.now_player_id = (pid + 1) % 3
                # 存在最优解，能瞬间出完牌，直接出牌不走模型，否则会导致梯度消失
                # 暂不考虑炸弹使奖惩翻番的效果，以胜利为唯一目标
                elif min([len(i) for i in tmp_act_ids]) == 1:
                    action = 0
                    for i in tmp_act_ids:
                        if len(i) == 1:
                            action = i[0]
                            break
                    env.step(1, id2combo[action])
                else:
                    h_pid = [(3 + player_id - pid) % 3 for player_id in history_pid]
                    h_vec = deepcopy(history_vec)

                    # 出牌
                    action, old_p = rl.f_act(state, act_ids, label_mask, attn_mask, dyn_vec, h_vec, h_pid)

                    if pid not in cache_memory:
                        cache_memory[pid] = []
                    cache_memory[pid].append((state, act_ids, label_mask, attn_mask, dyn_vec, action, h_vec, h_pid,
                                             old_p))

                    if action > 0:
                        history_vec.append(DynamicCorpus.get_combo_vec(id2combo[action]))
                        history_pid.append(pid)
                    env.step(action, id2combo[action])

        # PPO式学习
        if rl.f_pointer > rl.memory_size:
            f_loss = rl.learn_f()
        if (episode > 50000) and (step % 30 == 0):
            y_loss = rl.learn_y()
        if (episode > 50000) and (step % 30 == 0):
            e_loss = rl.learn_e()

        if episode % 1000 == 0 and episode != start_iter:
            rl.save_model('rl', episode)
            print("save: ", episode)
        if episode % 100 == 0:
            print("episode: ", episode, ", loss: ", y_loss, f_loss, e_loss, rl.e_pointer, rl.y_pointer)

# end of game
print('game over')
