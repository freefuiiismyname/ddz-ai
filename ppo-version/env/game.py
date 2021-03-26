# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from env.cards import *
from env.utils import *

from rl.common_func import sample
import numpy as np
import random
from copy import deepcopy

convert_dict = {-1: '不要', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: 'J', 10: 'Q', 11: 'K',
                12: 'A', 13: '2', 14: '小王', 15: '大王'}


# 玩家类
class Player(object):
    # 记录玩家信息
    def __init__(self, player_id):
        self.player_id = player_id
        self.cards_in_hand = []
        self.last_round_reward = 0


# 游戏类
class Game(object):
    def __init__(self):
        # 初始化一副扑克牌类
        self.cards = Cards()
        self.players = [Player(i) for i in range(3)]
        # play相关参数
        self.landlord_cards = None
        self.last_card_type = None
        self.last_rank = None
        self.last_move_player_id = None
        self.now_player_id = None
        self.landlord_count = None
        self.landlord_id = None
        self.keep_cards = None
        self.winner = None
        self.obs_landlord_cards = None
        self.last_move = None
        self.reset_time = 0

    def reset(self):
        """
        重置环境
        :return:
        """
        self.reset_time += 1
        np.random.shuffle(self.cards.cards)
        # 排序
        for i in range(3):
            cards = self.cards.cards[i * 17:(i + 1) * 17]
            cards.sort(key=lambda x: x.rank)
            self.players[i].cards_in_hand = cards

        self.landlord_cards = self.cards.cards[-3:]
        self.obs_landlord_cards = [0, 0, 0]
        self.winner = -1

        self.now_player_id = random.randint(0, 2)
        self.last_card_type = "lock"
        self.last_rank = 1
        self.last_move_player_id = self.now_player_id
        self.last_move = [0] * 15
        # 叫地主次数，三人均不叫时，重开游戏
        self.landlord_count = 0
        # 地主游戏id
        self.landlord_id = -1
        # 记录未打出的牌
        self.keep_cards = [4] * 13 + [1] * 2

    @staticmethod
    def search_whole_space(pid, cards, dc, up_cards, down_cards, position_id, sample_num=100):
        """
        全局化探索行动空间
        """
        # 宽度优先搜索
        act_space = bfs(cards)
        # 索引化搜索信息
        act_space_ids = act_space2ids_exactly(act_space, pid, dc, up_cards, down_cards)
        position_ids = [[position_id for _ in combos] for combos in act_space_ids if len(combos) > 0]
        act_space_ids = [combos for combos in act_space_ids if len(combos) > 0]
        attn_masks = [[1 for _ in combos] for combos in act_space_ids]
        # 采样
        act_space_ids, position_ids, attn_masks = sample(act_space_ids, position_ids, attn_masks, sample_num)
        return act_space_ids, position_ids, attn_masks

    def act_space_filter(self, act_space):
        """
        拆分可执行牌组集合和不可执行牌组集合
        """
        lm_record = {}
        res_as = []
        filter_as = []
        # 遍历牌组集合
        for a_s in act_space:
            trigger = False
            # 遍历牌组
            for a in a_s:
                # 判断牌组是否可以执行
                if str(a) not in lm_record:
                    _, card_type, rank = self.cards.check_legal_and_rank(a)
                    if card_type != self.last_card_type:
                        if card_type == '炸弹':
                            lm_record[str(a)] = 0
                        else:
                            lm_record[str(a)] = 1
                    else:
                        if rank > self.last_rank:
                            lm_record[str(a)] = 0
                        else:
                            lm_record[str(a)] = 1
                # 若可以执行，启动牌组集合的触发器
                if lm_record[str(a)] == 0:
                    trigger = True
            # 记录到可执行集合
            if trigger:
                res_as.append(a_s)
            # 记录到不可执行集合
            else:
                filter_as.append(a_s)
        return res_as, filter_as, lm_record

    def search_act_space(self, cards, dc, down_num, up_num, keep_cards, sample_num=100, bfs_only=True):
        """
        寻找玩家的动作空间
        """
        if bfs_only:
            # 宽度优先搜索
            act_space = bfs(cards)
            # 索引化搜索信息
            act_space_ids = act_space2ids(act_space, dc, down_num, up_num, keep_cards)
            label_masks = [[0 for _ in combos] for combos in act_space_ids if len(combos) > 0]
            act_space_ids = [combos for combos in act_space_ids if len(combos) > 0]
            attn_masks = [[1 for _ in combos] for combos in act_space_ids]
            # 采样
            act_space_ids, label_masks, attn_masks = sample(act_space_ids, label_masks, attn_masks, sample_num)
            return act_space_ids, label_masks, attn_masks
        else:
            """
            1.尝试从最优动作集中挑选行动
            """
            # 宽度优先搜索
            act_space = bfs(cards)
            res_as, filter_as, lm_record = self.act_space_filter(act_space)
            # 2.采样一部分可执行牌组集合，以及不可执行牌组集合（象征背后的机会成本）
            used_rate = len(res_as) / (len(res_as) + len(filter_as) + 0.001)
            # 对不可执行牌组集合的采样率cost rate，设置上限；避免对执行度的过遮掩
            cost_rate = min(1 - used_rate, 0.5)
            # 记录每类牌组集合的采样数
            k_num = min(len(res_as) + len(filter_as), sample_num)
            used_sample_num = int(k_num * used_rate)
            cost_sample_num = int(k_num * cost_rate)
            non_sample_num = sample_num - used_sample_num - cost_sample_num
            # 构成新的行动集
            res_as = random.sample(res_as, used_sample_num) + random.sample(filter_as, cost_sample_num)

            # 索引化搜索信息
            act_space_ids = act_space2ids(res_as, dc, down_num, up_num, keep_cards)
            label_masks = [[lm_record.get(str(i), 1) for i in a_s] for combos, a_s in zip(act_space_ids, res_as) if
                           len(combos) > 0]
            act_space_ids = [combos for combos in act_space_ids if len(combos) > 0]
            attn_masks = [[1 for _ in combos] for combos in act_space_ids]
            """
            2.如果未达到采样上限，利用有条件的宽度搜索寻找新采样
            """
            if non_sample_num > 0:
                # 记录最优动作集中已见牌组集合
                def sort_method(val):
                    return len(val) * 100 + val[0]

                seen_as = {}
                for a_s in res_as:
                    a_s.sort(key=sort_method)
                    seen_as[str(a_s)] = 1
                # 有条件的宽度优先搜索
                act_space = conditional_bfs(cards, self.last_card_type, self.last_rank, sum(self.last_move))
                # 过滤已见牌组集合
                res_as = []
                for a_s in act_space:
                    n_a_s = deepcopy(a_s)
                    n_a_s.sort(key=sort_method)
                    if str(n_a_s) in seen_as:
                        continue
                    res_as.append(a_s)
                # 对过滤后的可执行牌组集合进行索引化
                if len(res_as) > 0:
                    c_act_space_ids = act_space2ids(res_as, dc, down_num, up_num, keep_cards)

                    c_label_masks = [[0] + [1 for _ in combos[1:]] for combos in c_act_space_ids if len(combos) > 0]
                    c_act_space_ids = [combos for combos in c_act_space_ids if len(combos) > 0]
                    c_attn_masks = [[1 for _ in combos] for combos in c_act_space_ids]
                    # 余量采样
                    c_act_space_ids, c_label_masks, c_attn_masks = sample(c_act_space_ids, c_label_masks, c_attn_masks,
                                                                          non_sample_num)
                    # 将额外的可执行牌组集合加入到原集合中
                    act_space_ids = act_space_ids + c_act_space_ids
                    label_masks = label_masks + c_label_masks
                    attn_masks = attn_masks + c_attn_masks
            """
            3.每一个牌组集合增加【不出牌选项】
            """
            new_act_space_ids, new_label_masks, new_attn_masks = [], [], []
            for act_space_id, label_mask, attn_mask in zip(act_space_ids, label_masks, attn_masks):
                new_act_space_ids.append([0] + act_space_id)
                new_label_masks.append([0] + label_mask)
                new_attn_masks.append([1] + attn_mask)
            return new_act_space_ids, new_label_masks, new_attn_masks

    def observe_entirety(self, sample_num=100):
        """
        获取全面的观测信息
        """

        dyn_corpus = DynamicCorpus()
        all_act_space_ids, all_label_masks, all_attn_masks = [], [], []
        for player_id in range(3):
            # 与地主的相对位置
            position_id = (3 + player_id - self.now_player_id) % 3
            # 记录三位玩家的牌
            cards = []
            for card in self.players[player_id].cards_in_hand:
                cards.append(card.rank)
            down_cards = []
            for card in self.players[(player_id + 1) % 3].cards_in_hand:
                down_cards.append(card.rank)
            up_cards = []
            for card in self.players[(player_id + 2) % 3].cards_in_hand:
                up_cards.append(card.rank)
            # 探索该玩家的行动空间
            act_space_ids, label_masks, attn_masks = self.search_whole_space(player_id, cards, dyn_corpus, up_cards,
                                                                             down_cards, position_id, sample_num)
            # 记录玩家信息
            all_act_space_ids.append(act_space_ids)
            all_label_masks.append(label_masks)
            all_attn_masks.append(attn_masks)

        dyn_vec = dyn_corpus.id2vec

        return all_act_space_ids, dyn_vec, all_label_masks, all_attn_masks

    def observe(self, player_id, sample_num=100):
        """
        获取该玩家的观测信息
        """
        # self.show()

        reward = 0
        now_player = self.players[player_id]
        dyn_corpus = DynamicCorpus()

        # 整理上一轮奖励，地主和农民奖励一致，因为已经跳过叫地主环节
        if self.winner >= 0:
            if self.winner == self.landlord_id:
                if player_id == self.landlord_id:
                    reward += 1
                else:
                    reward -= 1
            else:
                if player_id == self.landlord_id:
                    reward -= 1
                else:
                    reward += 1

        # 整理玩家当前牌
        cards = []
        cards_in_hand = [0] * 15
        for card in now_player.cards_in_hand:
            cards.append(card.rank)
            cards_in_hand[card.rank - 1] += 1

        # 整理地主和出牌人信息
        last_move_player_rel_id = -1
        landlord_info = []
        for i in range(3):
            # 上一个出牌玩家id和所出牌，时局记录
            if self.last_move_player_id == (player_id + i) % 3:
                last_move_player_rel_id = i
            # 记录地主是谁
            if self.landlord_id == (player_id + i) % 3:
                landlord_info = [i]
        if not landlord_info:
            landlord_info = [-1]
        # 整理其他人手中余牌,各类牌型余牌数，每名玩家余牌数
        keep_cards = [j - i for i, j in zip(cards_in_hand, self.keep_cards)]
        down_num = len(self.players[(player_id + 1) % 3].cards_in_hand)
        up_num = len(self.players[(player_id + 2) % 3].cards_in_hand)
        other_card_info = keep_cards + [down_num, up_num]
        game_info = other_card_info + self.last_move + [last_move_player_rel_id]

        """
        动作空间探索
        """
        # 有牌权
        if last_move_player_rel_id == 0 or self.landlord_cards:
            act_space_ids, label_masks, attn_masks = self.search_act_space(cards, dyn_corpus, down_num, up_num,
                                                                           keep_cards, sample_num=sample_num)
        # 无牌权
        else:
            act_space_ids, label_masks, attn_masks = self.search_act_space(cards, dyn_corpus, down_num,
                                                                           up_num, keep_cards,
                                                                           sample_num=sample_num, bfs_only=False)

        dyn_vec = dyn_corpus.id2vec

        # 地主牌展示
        landlord_info += self.obs_landlord_cards
        state = cards_in_hand + game_info + landlord_info

        f_reward = now_player.last_round_reward + reward
        y_reward = reward
        # 为避免流局而降低表现，弊大于利
        # if self.landlord_count == 3:
        #     y_reward -= 0.1

        return state, f_reward, y_reward, act_space_ids, dyn_vec, dyn_corpus.id2combo, label_masks, attn_masks

    def campaign_for_landlord(self, is_campaign, player):
        """
        叫地主
        """
        move = [0] * 16
        # 叫地主
        if is_campaign:
            player.cards_in_hand += self.landlord_cards
            player.cards_in_hand.sort(key=lambda x: x.rank)
            self.landlord_id = self.now_player_id
            self.last_move_player_id = self.now_player_id
            self.obs_landlord_cards = [c.rank for c in self.landlord_cards]
            self.landlord_cards = None
            move[0] = 1
            # 不叫地主
        else:
            self.landlord_count += 1

    def fight(self, is_fight, player, out_cards):
        """
        出牌阶段
        """
        reward = 0
        move = [0] * 16

        # 选择【出牌】
        if is_fight > 0:
            # 检查牌型
            legal, self.last_card_type, self.last_rank = self.cards.check_legal_and_rank(out_cards)
            # 进行出牌更新
            new_cards = []
            out_num = 0
            for card in player.cards_in_hand:
                if out_num > len(out_cards) - 1:
                    new_cards.append(card)
                else:
                    if card.rank != out_cards[out_num]:
                        new_cards.append(card)
                    else:
                        out_num += 1
            # 将出牌转化为动作记录
            for card in out_cards:
                move[card] += 1
            self.last_move = move[1:]
            self.last_move_player_id = self.now_player_id
            self.keep_cards = [self.keep_cards[i] - self.last_move[i] for i in range(15)]
            player.cards_in_hand = new_cards

        # 当刚开始时，可以选择奖励出牌，加快模型收敛
        # player.last_round_reward = reward + sum(move) * 0.01
        # 训练至一定阶段，仅以胜利作为奖励
        player.last_round_reward = reward

    def step(self, act_type, out_cards=[], test_mode=False):
        """
        动作执行
        """
        now_player = self.players[self.now_player_id]
        # 抢地主环节
        if self.landlord_cards:
            self.campaign_for_landlord(is_campaign=act_type, player=now_player)
            if test_mode:
                print('玩家', self.now_player_id, '抢地主', act_type, '手上的牌',
                      ','.join([convert_dict[c.rank] for c in now_player.cards_in_hand]))
            if act_type == 0:
                self.now_player_id = (self.now_player_id + 1) % 3
        # 出牌环节
        else:
            self.fight(is_fight=act_type, player=now_player, out_cards=out_cards)
            # 判断游戏胜负
            if len(now_player.cards_in_hand) == 0:
                self.winner = self.now_player_id

            if test_mode:
                print('玩家', self.now_player_id, '出牌', ','.join([convert_dict[i] for i in out_cards]), '余牌',
                      ','.join([convert_dict[c.rank] for c in now_player.cards_in_hand]))
            self.now_player_id = (self.now_player_id + 1) % 3
