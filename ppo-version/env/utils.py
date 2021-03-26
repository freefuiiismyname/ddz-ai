# -*- coding: utf-8 -*-
# 动作空间探索
from __future__ import absolute_import
import itertools


def multiple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo, card_type, num=2):
    """
    连n
    """
    multiple = [leftover[0]]
    now_num = 1
    for i in leftover[1:]:
        if i > multiple[-1] + 1 or i > 12:
            break
        if i == multiple[-1] + 1:
            if now_num % num != 0:
                break
            multiple.append(i)
            now_num += 1
        else:
            if now_num % num != 0:
                multiple.append(i)
                now_num += 1
    if len(multiple) > 5:
        here_leftover = leftover[:]
        for j in range(5):
            pop_idx = here_leftover.index(multiple[j])
            here_leftover = here_leftover[:pop_idx] + here_leftover[pop_idx + 1:]

        for j in range(5, len(multiple)):
            pop_idx = here_leftover.index(multiple[j])
            here_leftover = here_leftover[:pop_idx] + here_leftover[pop_idx + 1:]

            if (j + 1) % num == 0:
                next_leftovers.append(here_leftover)
                next_card_nums.append(card_num - (j + 1))
                next_combos.append(combo + [[card_type] + multiple[:j + 1]])


def simple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo, card_type, combo_num=1):
    """
    k*n or 王炸
    """
    next_leftovers.append(leftover[combo_num:])
    next_card_nums.append(card_num - combo_num)
    next_combos.append(combo + [[card_type] + leftover[:combo_num]])


def smooth_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo, card_type):
    """
    顺子
    """
    smooth_cards = [leftover[0]]
    for i in leftover[1:]:
        if i > smooth_cards[-1] + 1 or i > 12:
            break
        if i == smooth_cards[-1] + 1:
            smooth_cards.append(i)

    if len(smooth_cards) > 4:
        here_leftover = leftover[:]
        for j in range(4):
            pop_idx = here_leftover.index(smooth_cards[j])
            here_leftover = here_leftover[:pop_idx] + here_leftover[pop_idx + 1:]

        for j in range(4, len(smooth_cards)):
            pop_idx = here_leftover.index(smooth_cards[j])
            here_leftover = here_leftover[:pop_idx] + here_leftover[pop_idx + 1:]
            next_leftovers.append(here_leftover)
            next_card_nums.append(card_num - (j + 1))
            next_combos.append(combo + [[card_type] + smooth_cards[:j + 1]])


def get_all_roads(carrier, single_cargo, double_cargo, now_roads=None, cargo_s_idx=1, allow_free=True):
    """
    获取所有可能的路径
    """
    if not carrier:
        return [i + single_cargo + double_cargo for i in now_roads]
    else:
        carr = carrier[0]
        if carr[0] == 2 or carr[0] == 6:
            # 因为card_type，所以减1
            carry_volume = int((len(carr) - 1) / 3)
        if carr[0] == 3 or carr[0] == 7:
            carry_volume = int((len(carr) - 1) / 4 * 2)
        # 三张2\炸弹3\飞机6\连四7
        # 不带
        all_roads = []
        if allow_free:
            if now_roads:
                here_roads = [i + [carr] for i in now_roads]
            else:
                here_roads = [[carr]]
            all_roads += get_all_roads(carrier[1:], single_cargo, double_cargo, here_roads)
        # 带单
        combinations = itertools.combinations(range(len(single_cargo)), carry_volume)
        for combination in combinations:
            s_carr = carr[:]
            keep_singe_cargo = []
            for idx, cargo in enumerate(single_cargo):
                if idx in combination:
                    s_carr += cargo[cargo_s_idx:]
                else:
                    keep_singe_cargo.append(cargo)

            if now_roads:
                here_roads = [i + [s_carr] for i in now_roads]
            else:
                here_roads = [[s_carr]]

            all_roads += get_all_roads(carrier[1:], keep_singe_cargo, double_cargo, here_roads)

        # 带双
        combinations = itertools.combinations(range(len(double_cargo)), carry_volume)
        for combination in combinations:
            d_carr = carr[:]
            keep_double_cargo = []
            for idx, cargo in enumerate(double_cargo):
                if idx in combination:
                    d_carr += cargo[cargo_s_idx:]
                else:
                    keep_double_cargo.append(cargo)
            if now_roads:
                here_roads = [i + [d_carr] for i in now_roads]
            else:
                here_roads = [[d_carr]]
            all_roads += get_all_roads(carrier[1:], single_cargo, keep_double_cargo, here_roads)

        # 双当作单带
        # if carry_volume % 2 == 0:
        #     carry_volume /= 2
        return all_roads


def index(arr, val):
    """
    索引，不存在则返回-1
    """
    try:
        idx = arr.index(val)
    except ValueError:
        idx = -1
    return idx


def get_latent_predator(cards, card_type, rank, last_num):
    """
    根据上一轮的牌权，寻找出牌序列
    """
    action_idx = []
    if card_type.startswith('单张'):
        seen_rank = []
        for j in range(len(cards) - 1, -1, -1):
            if cards[j] <= rank:
                break
            if cards[j] not in seen_rank:
                seen_rank.append(cards[j])
                action_idx.append([j])
    elif card_type.startswith('对子'):
        seen_rank = []
        for j in range(len(cards) - 1, 0, -1):
            if cards[j] <= rank:
                break
            if cards[j] not in seen_rank and cards[j] == cards[j - 1]:
                seen_rank.append(cards[j])
                action_idx.append([j, j - 1])
    elif card_type.startswith('顺子'):
        idx_map = [-1] * 12
        for idx, j in enumerate(cards):
            if j <= rank or j > 12:
                continue
            idx_map[j - 1] = idx
        # get_smooth_n(rank, last_num, idx_map, action_idx, 1)
        cursor = rank
        while cursor <= 12 - last_num:
            combo = idx_map[cursor:cursor + last_num]
            if -1 not in combo:
                action_idx.append(combo)
            cursor += 1
    elif card_type.startswith('连对'):
        idx_map = [-1] * 12 * 2
        for idx, j in enumerate(cards):
            if j <= rank or j > 12:
                continue
            if idx_map[2 * (j - 1)] == -1:
                idx_map[2 * (j - 1)] = idx
            else:
                idx_map[2 * (j - 1) + 1] = idx
        # get_smooth_n(rank, last_num, idx_map, action_idx, 2)
        cursor = rank
        while cursor <= 12 - last_num / 2:
            combo = idx_map[cursor * 2:cursor * 2 + last_num]
            if -1 not in combo:
                action_idx.append(combo)
            cursor += 1
    elif card_type.startswith('连三') or card_type.startswith('三张'):
        max_num = 12
        if last_num < 6:
            max_num = 13
        if "带一" in card_type:
            smooth_length = last_num * 3 / 4
        elif "带双" in card_type:
            smooth_length = last_num * 3 / 5
        else:
            smooth_length = last_num
        smooth_length = int(smooth_length)
        smooth_val = int(smooth_length / 3)
        triples = []
        idx_map = [-1] * 13 * 3
        for idx, j in enumerate(cards):
            if j <= rank or j > max_num:
                continue
            if idx_map[3 * (j - 1)] == -1:
                idx_map[3 * (j - 1)] = idx
            elif idx_map[3 * (j - 1) + 1] == -1:
                idx_map[3 * (j - 1) + 1] = idx
            else:
                idx_map[3 * (j - 1) + 2] = idx
        # get_smooth_n(rank, last_num, idx_map, action_idx, 3)
        cursor = rank
        while cursor <= max_num - smooth_val:
            combo = idx_map[cursor * 3:cursor * 3 + smooth_length]
            if -1 not in combo:
                triples.append(combo)
            cursor += 1
        if "带一" in card_type:
            for action in triples:
                single_count = [0] * 15
                singles = []
                for idx, i in enumerate(cards):
                    if single_count[i - 1] >= smooth_val:
                        continue
                    if idx not in action:
                        single_count[i - 1] += 1
                        singles.append([idx])
                if len(singles) >= smooth_val:
                    action_idx.append(action + [1] * smooth_val)
        elif "带双" in card_type:
            for action in triples:
                doubles = []
                not_used_cards = [(idx, i) for idx, i in enumerate(cards) if idx not in action]
                j = 0
                while j < len(not_used_cards) - 1:
                    idx, i = not_used_cards[j]
                    idx_, i_ = not_used_cards[j + 1]
                    if i == i_:
                        doubles.append([idx, idx_])
                        j += 2
                    else:
                        j += 1

                if len(doubles) >= smooth_val:
                    action_idx.append(action + [1, 1] * smooth_val)
        else:
            for action in triples:
                action_idx.append(action)
    elif card_type.startswith('连四') or card_type.startswith('炸弹'):
        max_num = 12
        if last_num < 8 or (last_num == 8 and "带两双" in card_type):
            max_num = 13

        if "带二" in card_type:
            smooth_length = last_num * 4 / 6
        elif "带两双" in card_type:
            smooth_length = last_num / 2
        else:
            smooth_length = last_num

        smooth_length = int(smooth_length)
        smooth_val = int(smooth_length / 4)
        quaternary = []
        idx_map = [-1] * 13 * 4
        for idx, j in enumerate(cards):
            if j <= rank or j > max_num:
                continue
            if idx_map[4 * (j - 1)] == -1:
                idx_map[4 * (j - 1)] = idx
            elif idx_map[4 * (j - 1) + 1] == -1:
                idx_map[4 * (j - 1) + 1] = idx
            elif idx_map[4 * (j - 1) + 2] == -1:
                idx_map[4 * (j - 1) + 2] = idx
            else:
                idx_map[4 * (j - 1) + 3] = idx
        # get_smooth_n(rank, last_num, idx_map, action_idx, 4)
        cursor = rank
        while cursor <= max_num - smooth_val:
            combo = idx_map[cursor * 4:cursor * 4 + smooth_length]
            if -1 not in combo:
                quaternary.append(combo)
            cursor += 1
        if "带二" in card_type:
            for action in quaternary:
                single_count = [0] * 15
                singles = []
                for idx, i in enumerate(cards):
                    if single_count[i - 1] >= smooth_val:
                        continue
                    if idx not in action:
                        single_count[i - 1] += 1
                        singles.append([idx])
                if len(singles) >= smooth_val * 2:
                    action_idx.append(action + [1] * smooth_val * 2)
        elif "带两双" in card_type:
            for action in quaternary:
                doubles = []
                not_used_cards = [(idx, i) for idx, i in enumerate(cards) if idx not in action]
                j = 0
                while j < len(not_used_cards) - 1:
                    idx, i = not_used_cards[j]
                    idx_, i_ = not_used_cards[j + 1]
                    if i == i_:
                        doubles.append([idx, idx_])
                        j += 2
                    else:
                        j += 1

                if len(doubles) >= smooth_val * 2:
                    action_idx.append(action + [1, 1] * smooth_val * 2)
        else:
            for action in quaternary:
                action_idx.append(action)
    if '炸弹' not in card_type:
        idx_map = {13: []}
        for idx, j in enumerate(cards):
            if j < 14:
                if j - 1 not in idx_map:
                    idx_map[j - 1] = []
                idx_map[j - 1].append(idx)
            else:
                idx_map[13].append(idx)
        if len(idx_map[13]) == 2:
            action_idx.append(idx_map[13])
        for v in idx_map.values():
            if len(v) == 4:
                action_idx.append(v)
    return action_idx


def conditional_bfs(cards, card_type, rank, last_num, gap=2, top_k=100):
    """
    根据上一轮的牌权，寻找出牌序列
    """
    action_idx = []
    if card_type.startswith('单张'):
        seen_rank = []
        for j in range(len(cards) - 1, -1, -1):
            if cards[j] <= rank:
                break
            if cards[j] not in seen_rank:
                seen_rank.append(cards[j])
                action_idx.append([j])
    elif card_type.startswith('对子'):
        seen_rank = []
        for j in range(len(cards) - 1, 0, -1):
            if cards[j] <= rank:
                break
            if cards[j] not in seen_rank and cards[j] == cards[j - 1]:
                seen_rank.append(cards[j])
                action_idx.append([j, j - 1])
    elif card_type.startswith('顺子'):
        idx_map = [-1] * 12
        for idx, j in enumerate(cards):
            if j <= rank or j > 12:
                continue
            idx_map[j - 1] = idx
        # get_smooth_n(rank, last_num, idx_map, action_idx, 1)
        cursor = rank
        while cursor <= 12 - last_num:
            combo = idx_map[cursor:cursor + last_num]
            if -1 not in combo:
                action_idx.append(combo)
            cursor += 1
    elif card_type.startswith('连对'):
        idx_map = [-1] * 12 * 2
        for idx, j in enumerate(cards):
            if j <= rank or j > 12:
                continue
            if idx_map[2 * (j - 1)] == -1:
                idx_map[2 * (j - 1)] = idx
            else:
                idx_map[2 * (j - 1) + 1] = idx
        # get_smooth_n(rank, last_num, idx_map, action_idx, 2)
        cursor = rank
        while cursor <= 12 - last_num / 2:
            combo = idx_map[cursor * 2:cursor * 2 + last_num]
            if -1 not in combo:
                action_idx.append(combo)
            cursor += 1
    elif card_type.startswith('连三') or card_type.startswith('三张'):
        max_num = 12
        if last_num < 6:
            max_num = 13
        if "带一" in card_type:
            smooth_length = last_num * 3 / 4
        elif "带双" in card_type:
            smooth_length = last_num * 3 / 5
        else:
            smooth_length = last_num
        smooth_length = int(smooth_length)
        smooth_val = int(smooth_length / 3)
        triples = []
        idx_map = [-1] * 13 * 3
        for idx, j in enumerate(cards):
            if j <= rank or j > max_num:
                continue
            if idx_map[3 * (j - 1)] == -1:
                idx_map[3 * (j - 1)] = idx
            elif idx_map[3 * (j - 1) + 1] == -1:
                idx_map[3 * (j - 1) + 1] = idx
            else:
                idx_map[3 * (j - 1) + 2] = idx
        # get_smooth_n(rank, last_num, idx_map, action_idx, 3)
        cursor = rank
        while cursor <= max_num - smooth_val:
            combo = idx_map[cursor * 3:cursor * 3 + smooth_length]
            if -1 not in combo:
                triples.append(combo)
            cursor += 1
        if "带一" in card_type:
            for action in triples:
                single_count = [0] * 15
                singles = []
                for idx, i in enumerate(cards):
                    if single_count[i - 1] >= smooth_val:
                        continue
                    if idx not in action:
                        single_count[i - 1] += 1
                        singles.append([idx])
                if len(singles) >= smooth_val:
                    roads = get_all_roads([[2] + action], singles, [], cargo_s_idx=0, allow_free=False)
                    if roads:
                        for road in roads:
                            action_idx.append(road[0][1:])
        elif "带双" in card_type:
            for action in triples:
                doubles = []
                not_used_cards = [(idx, i) for idx, i in enumerate(cards) if idx not in action]
                j = 0
                while j < len(not_used_cards) - 1:
                    idx, i = not_used_cards[j]
                    idx_, i_ = not_used_cards[j + 1]
                    if i == i_:
                        doubles.append([idx, idx_])
                        j += 2
                    else:
                        j += 1

                if len(doubles) >= smooth_val:
                    roads = get_all_roads([[2] + action], [], doubles, cargo_s_idx=0, allow_free=False)
                    if roads:
                        for road in roads:
                            action_idx.append(road[0][1:])
        else:
            for action in triples:
                action_idx.append(action)
    elif card_type.startswith('连四') or card_type.startswith('炸弹'):
        max_num = 12
        if last_num < 8 or (last_num == 8 and "带两双" in card_type):
            max_num = 13

        if "带二" in card_type:
            smooth_length = last_num * 4 / 6
        elif "带两双" in card_type:
            smooth_length = last_num / 2
        else:
            smooth_length = last_num

        smooth_length = int(smooth_length)
        smooth_val = int(smooth_length / 4)
        quaternary = []
        idx_map = [-1] * 13 * 4
        for idx, j in enumerate(cards):
            if j <= rank or j > max_num:
                continue
            if idx_map[4 * (j - 1)] == -1:
                idx_map[4 * (j - 1)] = idx
            elif idx_map[4 * (j - 1) + 1] == -1:
                idx_map[4 * (j - 1) + 1] = idx
            elif idx_map[4 * (j - 1) + 2] == -1:
                idx_map[4 * (j - 1) + 2] = idx
            else:
                idx_map[4 * (j - 1) + 3] = idx
        # get_smooth_n(rank, last_num, idx_map, action_idx, 4)
        cursor = rank
        while cursor <= max_num - smooth_val:
            combo = idx_map[cursor * 4:cursor * 4 + smooth_length]
            if -1 not in combo:
                quaternary.append(combo)
            cursor += 1
        if "带二" in card_type:
            for action in quaternary:
                single_count = [0] * 15
                singles = []
                for idx, i in enumerate(cards):
                    if single_count[i - 1] >= smooth_val:
                        continue
                    if idx not in action:
                        single_count[i - 1] += 1
                        singles.append([idx])
                if len(singles) >= smooth_val:
                    roads = get_all_roads([[3] + action], singles, [], cargo_s_idx=0, allow_free=False)
                    if roads:
                        for road in roads:
                            action_idx.append(road[0][1:])
        elif "带两双" in card_type:
            for action in quaternary:
                doubles = []
                not_used_cards = [(idx, i) for idx, i in enumerate(cards) if idx not in action]
                j = 0
                while j < len(not_used_cards) - 1:
                    idx, i = not_used_cards[j]
                    idx_, i_ = not_used_cards[j + 1]
                    if i == i_:
                        doubles.append([idx, idx_])
                        j += 2
                    else:
                        j += 1

                if len(doubles) >= smooth_val:
                    roads = get_all_roads([[3] + action], [], doubles, cargo_s_idx=0, allow_free=False)
                    if roads:
                        for road in roads:
                            action_idx.append(road[0][1:])
        else:
            for action in quaternary:
                action_idx.append(action)
    if '炸弹' not in card_type:
        idx_map = {13: []}
        for idx, j in enumerate(cards):
            if j < 14:
                if j - 1 not in idx_map:
                    idx_map[j - 1] = []
                idx_map[j - 1].append(idx)
            else:
                idx_map[13].append(idx)
        if len(idx_map[13]) == 2:
            action_idx.append(idx_map[13])
        for v in idx_map.values():
            if len(v) == 4:
                action_idx.append(v)

    available_actions = []
    for idx in action_idx:
        action = []
        keep_cards = []
        for i, card in enumerate(cards):
            if i in idx:
                action.append(card)
            else:
                keep_cards.append(card)
        available_actions += [[action] + i for i in bfs(keep_cards)]
    min_step = min([999] + [len(actions) for actions in available_actions])

    for j in range(5, gap - 1, -1):
        if len(available_actions) < top_k:
            break
        available_actions = [i for i in available_actions if len(i) <= min_step + j]

    return available_actions


def get_smooth_n(rank, last_num, idx_map, action_idx, n):
    """
    比-1 not in idx_map快
    """
    cursor = rank
    smooth_num = 0
    while cursor <= 12 - last_num / n:
        combo = idx_map[cursor * n:cursor * n + last_num]
        for j in range(last_num - 1, smooth_num - 1, -1):
            if combo[j] == -1:
                cursor += int(j / n)
                smooth_num = last_num - 1 - j
                smooth_num = smooth_num - smooth_num % n
                break
            smooth_num += 1
        if smooth_num == last_num:
            smooth_num -= n
            action_idx.append(combo)
        else:
            smooth_num = smooth_num - smooth_num % n
        cursor += 1


def bfs(cards, top_k=100, gap=2):
    """
    宽度优先搜索
    """
    leftovers = [cards]
    card_nums = [len(cards)]
    combos = [[]]
    top_count = 0
    while max(card_nums) > 0 and top_count < top_k:
        next_leftovers = []
        next_card_nums = []
        next_combos = []
        for leftover, card_num, combo in zip(leftovers, card_nums, combos):
            if card_num == 0:
                next_leftovers.append(leftover)
                next_card_nums.append(card_num)
                next_combos.append(combo)
                continue
            # 单张
            simple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo, card_type=0,
                         combo_num=1)
            if card_num > 1:
                # 对子 或 王炸
                if leftover[0] == leftover[1]:
                    simple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo,
                                 card_type=1, combo_num=2)
                if leftover[0] > 13:
                    simple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo,
                                 card_type=3, combo_num=2)
            if card_num > 2:
                # 三张
                if leftover[0] == leftover[2]:
                    simple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo,
                                 card_type=2, combo_num=3)
            if card_num > 3:
                # 四张
                if leftover[0] == leftover[3]:
                    simple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo,
                                 card_type=3, combo_num=4)
            if card_num > 4:
                # 顺子
                smooth_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo, card_type=4)
            if card_num > 5:
                # 连对
                multiple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo, card_type=5,
                               num=2)
                # 飞机
                multiple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo, card_type=6,
                               num=3)
                # 连四
                multiple_combo(next_leftovers, next_card_nums, next_combos, leftover, card_num, combo, card_type=7,
                               num=4)
        leftovers = next_leftovers
        card_nums = next_card_nums
        combos = next_combos
        top_count = sum([1 for card_num in card_nums if card_num == 0])
    combos = [combo for combo, card_num in zip(combos, card_nums) if card_num == 0]

    combos.sort(key=len)
    # 去重
    seen_dict = {}
    good_combos = []
    for combo in combos:
        combo.sort(key=len)
        if str(combo) not in seen_dict:
            seen_dict[str(combo)] = 0
            good_combos.append(combo)

    # 带牌组合探索
    final_combos = []
    for combo in good_combos:
        # 寻找三张2\炸弹3\飞机6\连四7
        carrier = [cards for cards in combo if cards[0] in [2, 3, 6, 7] and len(cards) > 3]
        if carrier:
            singe_cargo = [cards for cards in combo if cards[0] == 0]
            double_cargo = [cards for cards in combo if cards[0] == 1]
            remnant = [cards for cards in combo if cards[0] in [4, 5]] + [cards for cards in combo if
                                                                          cards[0] in [3] and len(cards) <= 3]
            # Note并不能覆盖所有情况，不过一般能获得最优解
            roads = get_all_roads(carrier, singe_cargo, double_cargo)
            roads = [i + remnant for i in roads]
            final_combos += roads
    # 清洗
    seen_dict = {}
    final_combos += good_combos
    res_combos = []
    for combo in final_combos:
        combo = [i[1:] for i in combo]
        combo.sort(key=len)
        if str(combo) not in seen_dict:
            seen_dict[str(combo)] = 0
            res_combos.append(combo)

    combo_steps = [len(combo) for combo in res_combos]
    min_step = min([9999] + combo_steps)
    for j in range(5, gap - 1, -1):
        if len(res_combos) < top_k:
            break
        res_combos = [i for i in res_combos if len(i) <= min_step + j]
    return res_combos


if __name__ == "__main__":
    conditional_bfs([1, 1, 2, 2, 3, 4, 4, 6, 6, 7,  8, 9, 10, 10, 11, 11, 11, 12, 13, 14, 15], '单张', 1, 1)
    res = bfs([1, 1, 2, 2, 3, 4, 4, 6, 6, 7, 8, 9, 10, 10, 11, 11, 11, 12, 13, 14, 15])
    print(len([1, 1, 2, 2, 3, 4, 4, 6, 6, 7, 8, 9, 10, 10, 11, 11, 11, 13]))
    res = [i[0] for i in res if i[0][0] == 3]
    print(len(res))
    print(res)
