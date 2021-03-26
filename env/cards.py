# -*- coding: utf-8 -*-
"""
扑克牌卡片规则
"""
import numpy as np
from collections import Counter
from env.utils import get_latent_predator

CARD_TYPE = ['炸弹', '单张', '对子', '顺子', '连对', '连三', '连三带一', '连三带双', '连四', '连四带二', '连四带两双']


def act_space2ids_exactly(act_space, pid, dyn_corpus, up_cards, down_cards):
    """
    精确地将combo转化为动态嵌入表的id
    """
    act_space_ids = []
    # 遍历每一个牌组集合
    for combos in act_space:
        combo_ids = []
        # 遍历牌组
        for combo in combos:
            # 获取牌组id 和 记录牌组向量
            _, card_type, rank = Cards.check_legal_and_rank(combo)
            card_type_id = CARD_TYPE.index(card_type.split('-')[0])
            combo_id = dyn_corpus.get_combo_id_exactly(combo, pid, card_type, card_type_id, rank, up_cards, down_cards)
            combo_ids.append(combo_id)
        act_space_ids.append(combo_ids)
    return act_space_ids


def act_space2ids(act_space, dyn_corpus, down_num, up_num, keep_cards):
    """
    将combo转化为动态嵌入表的id
    """
    # 转换外部余牌
    cards = []
    for i, j in enumerate(keep_cards):
        for _ in range(j):
            cards.append(i + 1)

    # 遍历每一个牌组集合
    act_space_ids = []
    for combos in act_space:
        # 遍历牌组
        combo_ids = []
        for combo in combos:
            # 获取牌组id 和 记录牌组向量
            _, card_type, rank = Cards.check_legal_and_rank(combo)
            card_type_id = CARD_TYPE.index(card_type.split('-')[0])
            combo_id = dyn_corpus.get_combo_id(combo, card_type, card_type_id, rank, down_num, up_num, cards)
            combo_ids.append(combo_id)

        act_space_ids.append(combo_ids)
    return act_space_ids


class Cards(object):
    """
    一副扑克牌类,54张排,abcd四种花色,小王14-a,大王15-a
    """

    def __init__(self):
        # 初始化扑克牌类型
        self.cards_type = ['A-12'] * 4 + ['2-13'] * 4 + ['3-1'] * 4 + ['4-2'] * 4 + ['5-3'] * 4 + ['6-4'] * 4 + \
                          ['7-5'] * 4 + ['8-6'] * 4 + ['9-7'] * 4 + ['10-8'] * 4 + ['J-9'] * 4 + ['Q-10'] * 4 + \
                          ['K-11'] * 4 + ['小王-14', '大王-15']

        # 初始化扑克牌类
        self.cards = self.get_cards()

    # 初始化扑克牌类
    def get_cards(self):
        cards = []
        for card_type in self.cards_type:
            cards.append(Card(card_type))
        # 打乱顺序
        np.random.shuffle(cards)
        return cards

    @staticmethod
    def is_bigger(bef_card_type, bef_rank, now_card_type, now_rank):
        """
        比较两手牌的大小
        """
        if bef_card_type == 'lock':
            return True
        if bef_card_type != now_card_type:
            if now_card_type == '炸弹':
                return True
            return False
        if now_rank > bef_rank:
            return True
        return False

    @staticmethod
    def check_triple(out_cards, triples, card_map, quaternary):
        """
        连三情况
        """
        triples.sort()

        diff = [sum([1 + j * 0.01 for j in range(i, -1, -1) if i - j == triples[i] - triples[j]]) for
                i in range(len(triples) - 1, -1, -1)][::-1]
        best_idx, best_num = np.argmax(diff), int(max(diff))
        triples = [triples[i] for i in range(len(triples)) if best_idx - best_num < i <= best_idx]

        if len(out_cards) - 3 * len(triples) == 0:
            diff = [triples[i + 1] - triples[i] - 1 for i in range(len(triples) - 1)]
            if sum(diff) == 0:
                return True, '连三-' + str(len(out_cards)) + '张', triples[0]
            single_num = len([1 for i in diff if i > 0])
            used_triples = [j for i, j in zip(diff + [1], triples) if i == 0]
            if (len(triples) - single_num) / single_num == 3:
                return True, '连三带一-' + str(len(out_cards)) + '张', used_triples[0]
        if len(out_cards) - 3 * len(triples) == len(triples):
            diff = [triples[i + 1] - triples[i] - 1 for i in range(len(triples) - 1)]

            if sum(diff) == 0:
                return True, '连三带一-' + str(len(out_cards)) + '张', triples[0]
        doubles = [k for k, v in card_map.items() if v == 2]
        if len(out_cards) - 3 * len(triples) == len(doubles) * 2 + len(quaternary) * 4 and len(
                doubles) + len(quaternary) * 2 == len(triples):
            diff = [triples[i + 1] - triples[i] - 1 for i in range(len(triples) - 1)]
            if sum(diff) == 0:
                return True, '连三带双-' + str(len(out_cards)) + '张', triples[0]
        return False, '不合规', -1

    @staticmethod
    def check_legal_and_rank(out_cards, last_card_type=''):
        """
        last_card_type 解决出牌的歧义性问题，备用
        #TODO [11, 11, 11, 11, 12, 12, 12, 12]，可理解4*12和4*11，或4*12和两个2*11
        第一阶段游戏规则允许歧义性改变
        检查出牌的合理性
        """
        out_cards.sort()
        legal, card_type, rank = False, '不合规', -1
        card_map = Counter(out_cards)

        # 单张
        if len(out_cards) == 1:
            return True, '单张', out_cards[0]
        # 双
        elif len(out_cards) == 2:
            if len(card_map.keys()) == 1:
                return True, '对子', out_cards[0]
            elif [14, 15] == out_cards:
                return True, '炸弹', out_cards[0]
        # 三张
        elif len(out_cards) == 3:
            if len(card_map.keys()) == 1:
                return True, '连三', out_cards[0]
        # 三带一、炸弹
        elif len(out_cards) == 4:
            if len(card_map.keys()) == 1:
                return True, '炸弹', out_cards[0]
            else:
                triple = [k for k, v in card_map.items() if v == 3]
                if triple:
                    return True, '连三带一', triple[0]
        # 顺子、连对、飞机、四带二、八带四、三带二
        else:
            if out_cards[-1] < 13:
                # 顺子
                diff = [out_cards[i + 1] - out_cards[i] - 1 for i in range(len(out_cards) - 1)]
                if sum(diff) == 0 and -1 not in diff:
                    return True, '顺子-' + str(len(out_cards)) + '张', out_cards[0]
                # 连对
                if len([v for v in card_map.values() if v != 2]) == 0:
                    diff_cards = out_cards[::2]
                    diff = [diff_cards[i + 1] - diff_cards[i] - 1 for i in range(len(diff_cards) - 1)]
                    if sum(diff) == 0 and -1 not in diff:
                        return True, '连对-' + str(len(out_cards)) + '张', out_cards[0]
            # 三带二
            if 3 in card_map.values() and 2 == len(card_map.values()):
                return True, '连三带双', [k for k, v in card_map.items() if v == 3][0]

            # 飞机：不带，每三带单，每三带双
            triples_a = [k for k, v in card_map.items() if v >= 3 and k < 13]
            triples_b = [k for k, v in card_map.items() if v == 3 and k < 13]
            quaternary = [k for k, v in card_map.items() if v == 4]
            # 无法连到2以上
            if triples_a:
                legal, card_type, rank = Cards.check_triple(out_cards, triples_a, card_map, quaternary)
                if legal:
                    return legal, card_type, rank

            if triples_b and len(triples_a) > len(triples_b):
                legal, card_type, rank = Cards.check_triple(out_cards, triples_b, card_map, quaternary)
                if legal:
                    return legal, card_type, rank

            # 大飞机：不带，每四带二，每四带两双
            # 寻找大飞机带炸弹的最大价值方案
            if quaternary:
                quaternary.sort()
                if quaternary[-1] < 13 or len(out_cards) < 9:
                    diff = [sum([1 + j * 0.01 for j in range(i, -1, -1) if i - j == quaternary[i] - quaternary[j]]) for
                            i in
                            range(len(quaternary) - 1, -1, -1)][::-1]
                    best_idx, best_num = np.argmax(diff), int(max(diff))
                    pop_quaternary = [quaternary[i] for i in range(len(quaternary)) if
                                      i > best_idx or i <= best_idx - best_num]
                    quaternary = [quaternary[i] for i in range(len(quaternary)) if
                                  best_idx - best_num < i <= best_idx]
                    if len(out_cards) - 4 * len(quaternary) == 0:
                        diff = [quaternary[i + 1] - quaternary[i] - 1 for i in range(len(quaternary) - 1)]
                        if sum(diff) == 0:
                            return True, '连四-' + str(len(out_cards)) + '张', quaternary[0]
                    if len(out_cards) - 4 * len(quaternary) == 2 * len(quaternary):
                        diff = [quaternary[i + 1] - quaternary[i] - 1 for i in range(len(quaternary) - 1)]
                        if sum(diff) == 0:
                            return True, '连四带二-' + str(len(out_cards)) + '张', quaternary[0]
                    doubles = [k for k, v in card_map.items() if v == 2]
                    if len(out_cards) - 4 * len(quaternary) == len(doubles) * 2 + len(pop_quaternary) * 4 and len(
                            doubles) + len(pop_quaternary) * 2 == 2 * len(quaternary):
                        diff = [quaternary[i + 1] - quaternary[i] - 1 for i in range(len(quaternary) - 1)]
                        if sum(diff) == 0:
                            return True, '连四带两双-' + str(len(out_cards)) + '张', quaternary[0]

        return legal, card_type, rank


class Card(object):
    """
    扑克牌类
    """

    def __init__(self, card_type):
        self.card_type = card_type
        # 名称
        self.name = self.card_type.split('-')[0]
        # 大小
        self.rank = int(self.card_type.split('-')[1])

    def __repr__(self):
        # return self.name
        return str(self.rank)


class DynamicCorpus(object):
    """
    由于牌型演化空间大，以动态牌向量表作为核心，以便copy机制的实现
    """

    def __init__(self):
        self.combo2id = {}
        self.id2combo = []
        self.id2vec = []
        # <不要>或<不叫地主>
        self.combo2id['-1'] = len(self.id2vec)
        self.id2combo.append([-1])
        self.id2vec.append([-1, 0, 0, 0, 0])

    @staticmethod
    def get_combo_vec(combo):
        """
        获取combo对应的vec
        """
        _, card_type, rank = Cards.check_legal_and_rank(combo)
        card_type_id = CARD_TYPE.index(card_type.split('-')[0])
        return [card_type_id, rank, len(combo), 0, 0]

    def get_combo_id(self, combo, card_type, card_type_id, rank, down_num, up_num, cards):
        """
        TODO avg_bring 让模型知道它携带牌的大小
        获取combo对应的id，并视情况更新动态嵌入表
        """
        combo.sort()
        tag = ','.join([str(i) for i in combo])
        if tag not in self.combo2id:
            # TODO 可以考虑地主牌，进一步得到精确结果
            predators = get_latent_predator(cards, card_type, rank, len(combo))
            up = len([i for i in predators if len(i) <= up_num])
            dp = len([i for i in predators if len(i) <= down_num])

            self.combo2id[tag] = len(self.id2vec)
            self.id2combo.append(combo)
            self.id2vec.append([card_type_id, rank, len(combo), up, dp])
        return self.combo2id[tag]

    def get_combo_id_exactly(self, combo, pid, card_type, card_type_id, rank, up_cards, down_cards):
        """
        精确地获取combo对应的id，并视情况更新动态嵌入表
        """
        combo.sort()
        tag = ','.join([str(i) for i in combo]) + ':' + str(pid)
        if tag not in self.combo2id:
            up = len(get_latent_predator(up_cards, card_type, rank, len(combo)))
            dp = len(get_latent_predator(down_cards, card_type, rank, len(combo)))

            self.combo2id[tag] = len(self.id2vec)
            self.id2combo.append(combo)
            self.id2vec.append([card_type_id, rank, len(combo), up, dp])
        return self.combo2id[tag]
