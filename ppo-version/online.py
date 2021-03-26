# -*- coding: utf-8 -*-
from __future__ import absolute_import
from env.online_game import Game
from rl.modeling import RL
from copy import deepcopy
from env.cards import DynamicCorpus


def convert_cards(_cards):
    """
    将字符串转换为牌列表
    """
    _cards = _cards.strip().replace(' ', '').replace('，', ',')
    _cards = [int(i) for i in _cards.split(',')]
    return _cards


if __name__ == "__main__":
    start_iter = 83000
    init_checkpoint = None
    num_epochs = 2000001
    dim_states = 52
    rl = RL(dim_states, lr_a=0.0001, lr_c=0.0001, init_checkpoint=init_checkpoint)

    # fine-tune
    if start_iter != 0 and not init_checkpoint:
        rl.load_model('rl', start_iter)
    env = Game()
    for episode in range(start_iter, num_epochs):
        # 输入手上的牌
        # 输入第一名抢地主的玩家是谁
        print('新局开始，输入牌规则介绍：1.大小从1-15，1代表3，14代表小王，15代表大王,逗号分隔如【1,2,3,4,5】；2.任何出牌阶段输入-2，表示游戏结束')
        pid = int(input('请输入谁先叫地主，2代表上家，0代表自己，1代表下家').strip().replace(' ', ''))
        cards = input('请输入手中的牌')
        env.reset(convert_cards(cards), pid)
        # 体验采样，由评估器分析此局是否值得进行游戏
        history_vec = []
        history_pid = []
        while 1:
            """
            1. 使用模型推荐方案
            """
            if pid == 0:
                state, act_ids, dyn_vec, id2combo, label_mask, attn_mask = env.observe()
                # 叫地主环节
                if env.landlord_cards:
                    action = rl.y_act(state, act_ids, attn_mask, dyn_vec)
                    # 更新记忆器
                    if action > 0:
                        print('建议：：：抢地主')
                    else:
                        print('建议：：：不抢地主')

                # 斗地主环节
                else:
                    # 出不了牌
                    tmp_act_ids = [[j for j, lmm in zip(i, lm) if j > 0 and lmm == 0] for i, lm in
                                   zip(act_ids, label_mask)]
                    tmp_label_mask = [[lmm for j, lmm in zip(i, lm) if j > 0 and lmm == 0] for i, lm in
                                      zip(act_ids, label_mask)]
                    if max([len(i) for i in tmp_label_mask] + [0]) == 0:
                        print('建议：：：无法出牌')
                    # 存在最优解，能瞬间出完牌，直接出牌不走模型，否则会导致梯度消失
                    # 暂不考虑炸弹使奖惩翻番的效果，以胜利为唯一目标
                    elif min([len(i) for i in tmp_act_ids]) == 1:
                        action = 0
                        for i in tmp_act_ids:
                            if len(i) == 1:
                                action = i[0]
                                break
                        print('建议：：：', id2combo[action])
                    else:
                        h_pid = [(3 + player_id - pid) % 3 for player_id in history_pid]
                        h_vec = deepcopy(history_vec)
                        # 出牌
                        action, a_val = rl.f_act(state, act_ids, label_mask, attn_mask, dyn_vec, h_vec, h_pid)
                        print('建议：：：', id2combo[action])
            """
            2. 输入玩家动作
            """
            if env.landlord_cards:
                action = input('玩家' + str(pid) + '【叫地主？】，-1代表不叫地主,否则输入地主牌')
                if '-2' in action:
                    break
                if '-1' not in action:
                    env.step_yield(pid, convert_cards(action))
                    history_vec.append([-1, -1, -1, -1, -1])
                    history_pid.append(pid)
                else:
                    pid = (pid + 1) % 3
            else:
                action = input('玩家' + str(pid) + '【出牌？】，-1代表不出牌,否则输入所出牌')
                if '-2' in action:
                    break
                if '-1' not in action:
                    action = convert_cards(action)
                    env.step_fight(pid, action)
                    history_vec.append(DynamicCorpus.get_combo_vec(action))
                    history_pid.append(pid)
                pid = (pid + 1) % 3
