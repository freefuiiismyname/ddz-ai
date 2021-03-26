# -*- coding: utf-8 -*-
from __future__ import absolute_import
from env.game import Game
from rl.modeling import RL
from copy import deepcopy
from env.cards import DynamicCorpus

if __name__ == "__main__":

    start_iter = 50000
    init_checkpoint = None
    num_epochs = 2000001
    dim_states = 52
    rl = RL(dim_states, lr_a=0.0001, lr_c=0.0001, init_checkpoint=init_checkpoint)

    # fine-tune
    if start_iter != 0 and not init_checkpoint:
        rl.load_model('rl', start_iter)

    env = Game()
    for episode in range(start_iter, num_epochs):
        env.reset()
        print()

        history_vec = []
        history_pid = []
        while 1:
            pid = env.now_player_id
            # 无人叫地主 or 游戏结束，记录所有存档
            if env.landlord_count == 3 or env.winner >= 0:
                for i in range(3):
                    state, f_reward, y_reward, act_ids, dyn_vec, _, label_mask, attn_mask = env.observe(pid)
                    print('玩家', pid, '获得奖励', y_reward)
                    pid = (pid + 1) % 3
                    env.now_player_id = pid
                break

            state, f_reward, _, act_ids, dyn_vec, id2combo, label_mask, attn_mask = env.observe(pid)

            # 叫地主环节
            if env.landlord_cards:
                action = rl.y_act(state, act_ids, attn_mask, dyn_vec, test_mode=True)

                # 更新记忆器
                env.step(action, test_mode=True)
                if action > 0:
                    history_vec.append([-1, -1, -1, -1, -1])
                    history_pid.append(pid)
                    e_act_ids, e_dyn_vec, e_label_mask, e_attn_mask = env.observe_entirety()
                    # 判断是否值得游戏
                    e_action, next_epi = rl.e_act(e_act_ids, e_dyn_vec, e_label_mask, e_attn_mask, test_mode=True)


            # 斗地主环节
            else:
                # 出不了牌
                tmp_act_ids = [[j for j, lmm in zip(i, lm) if j > 0 and lmm == 0] for i, lm in zip(act_ids, label_mask)]
                tmp_label_mask = [[lmm for j, lmm in zip(i, lm) if j > 0 and lmm == 0] for i, lm in
                                  zip(act_ids, label_mask)]
                if max([len(i) for i in tmp_label_mask] + [0]) == 0:
                    env.now_player_id = (pid + 1) % 3
                # 存在最优解，能瞬间出完牌，直接出牌不走模型，否则会导致梯度消失
                # 暂不考虑炸弹使奖惩翻番的效果，以胜利为唯一目标
                elif min([len(i) for i in tmp_act_ids]) == 1:
                    action = 0
                    for i in tmp_act_ids:
                        if len(i) == 1:
                            action = i[0]
                            break
                    env.step(1, id2combo[action], test_mode=True)
                else:
                    h_pid = [(3 + player_id - pid) % 3 for player_id in history_pid]
                    h_vec = deepcopy(history_vec)
                    # 出牌
                    action, a_val = rl.f_act(state, act_ids, label_mask, attn_mask, dyn_vec, h_vec, h_pid)
                    if action > 0:
                        history_vec.append(DynamicCorpus.get_combo_vec(id2combo[action]))
                        history_pid.append(pid)
                    env.step(action, id2combo[action], test_mode=True)
