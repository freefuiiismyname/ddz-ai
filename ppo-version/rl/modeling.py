# -*- coding: utf-8 -*-
from rl.transformer import *
from rl.common_func import *


# AI模型类
class RL(object):
    def __init__(self, s_dim, lr_a=0.0001, lr_c=0.0002, reward_decay=0.95, memory_size=120, batch_size=12, dim_size=5,
                 init_checkpoint=None, state_num=10):
        self.sess = tf.Session()
        self.f_memory, self.y_memory, self.e_memory = {}, {}, {}
        self.f_pointer, self.y_pointer, self.e_pointer = 0, 0, 0

        self.state_num = state_num

        self.s_dim, self.reward_decay, self.lr_a, self.lr_c = s_dim, reward_decay, lr_a, lr_c
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dim_size = dim_size

        self.act_ids = tf.placeholder(tf.int32, [None, None, None], 'a_s')
        self.act_nd_ids = tf.placeholder(tf.int32, [None, None, None, 2], 'a_s_nd')
        self.d_v = tf.placeholder(tf.float32, [None, None, self.dim_size], 'd_v')
        self.e_cls = tf.placeholder(tf.int32, [None], 'e_label')
        self.y_l_cls = tf.placeholder(tf.int32, [None], 'y_l_label')
        self.y_f_cls = tf.placeholder(tf.int32, [None], 'y_f_label')
        self.loss_mask = tf.placeholder(tf.float32, [None, 2], 'loss_mask')
        vocab_size = get_shape_list(self.d_v)[1]
        act_vec = tf.gather_nd(self.d_v, self.act_nd_ids)

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.l_m = tf.placeholder(tf.int32, [None, None, None], 'l_m')
        self.a_m = tf.placeholder(tf.float32, [None, None, None], 'a_m')
        self.h_vec = tf.placeholder(tf.float32, [None, None, self.dim_size], 'h_v')
        self.h_pid = tf.placeholder(tf.int32, [None, None], 'h_p')
        self.h_m = tf.placeholder(tf.float32, [None, None], 'h_m')

        self.f_action = tf.placeholder(tf.int32, [None, 2], 'f_a')
        self.y_action = tf.placeholder(tf.int32, [None, 2], 'y_a')
        self.e_action = tf.placeholder(tf.int32, [None, 2], 'e_a')

        self.old_p = tf.placeholder(tf.float32, [None], 'old_p')
        """
        yield model
        """
        with tf.variable_scope('Y/Actor'):
            self.y_l, self.y_f = self.build_y(act_vec, self.S, self.a_m)
            l_one_hot_labels = tf.one_hot(self.y_l_cls, depth=2, dtype=tf.float32)
            l_log_probs = tf.log(self.y_l)
            y_l_loss = tf.reduce_sum(-tf.reduce_sum(l_one_hot_labels * l_log_probs, axis=1) * self.loss_mask[:, 1],
                                     axis=0)
            f_one_hot_labels = tf.one_hot(self.y_f_cls, depth=2, dtype=tf.float32)
            f_log_probs = tf.log(self.y_f)
            y_f_loss = tf.reduce_sum(-tf.reduce_sum(f_one_hot_labels * f_log_probs, axis=1) * self.loss_mask[:, 0],
                                     axis=0)

        self.y_loss = (y_l_loss + y_f_loss) / self.batch_size

        self.y_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Y/Actor')
        y_optimizer = tf.train.AdamOptimizer(self.lr_a)
        self.y_train = clip_train_op(self.y_loss, self.y_params, y_optimizer)

        """
        evaluate model
        """
        with tf.variable_scope('E/Actor'):
            self.e = self.build_e(act_vec, self.l_m, self.a_m)
            one_hot_labels = tf.one_hot(self.e_cls, depth=2, dtype=tf.float32)
            log_probs = tf.log(self.e)
        self.e_loss = tf.reduce_mean(-tf.reduce_sum(one_hot_labels * log_probs, axis=1))

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='E/Actor')
        e_optimizer = tf.train.AdamOptimizer(self.lr_a)
        self.e_train = clip_train_op(self.e_loss, self.e_params, e_optimizer)

        """
        fight model
        """
        with tf.variable_scope('F'):
            with tf.variable_scope('Actor'):
                self.a = self.build_a(act_vec, self.S, self.a_m, self.l_m, vocab_size, self.act_ids, self.h_vec,
                                      self.h_pid, self.h_m)
            with tf.variable_scope('Critic'):
                with tf.variable_scope('eval'):
                    q = self.build_c(act_vec, self.S, self.a_m, self.l_m, self.h_vec, self.h_pid, self.h_m)

        self.a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='F/Actor')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='F/Critic/eval')

        self.adv = self.R - q
        self.c_loss = tf.reduce_mean(tf.square(self.adv))

        c_optimizer = tf.train.AdamOptimizer(self.lr_c)
        self.c_train = clip_train_op(self.c_loss, self.ce_params, c_optimizer)

        self.acts_prob = tf.reshape(tf.gather_nd(self.a, self.f_action), [-1, 1])
        ratio = self.acts_prob/(self.old_p + 1e-6)
        surr = ratio * self.adv
        self.a_loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 0.8, 1.2) * self.adv))

        # dist = tf.distributions.Categorical(probs=self.a+1e-9) # 由于label mask，需要加epsilon
        # entropy = tf.reduce_mean(dist.entropy())
        # self.a_loss = self.a_loss - 0.01*entropy
        a_optimizer = tf.train.AdamOptimizer(self.lr_a)
        self.a_train = clip_train_op(self.a_loss, self.a_params, a_optimizer)

        # 对模型额外修改时使用
        if init_checkpoint:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        self.sess.run(tf.global_variables_initializer())

    def embedding_postprocessor(self, token_type_ids, batch_size, combo_num, combo_length, dim_size, type_size=2):
        """
        动作类型嵌入
        """
        token_type_table = tf.get_variable(name='label_tag', shape=[type_size, self.dim_size],
                                           initializer=create_initializer())
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=type_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, combo_num, combo_length, dim_size])
        return token_type_embeddings

    def hns_encoder(self, h_vec, h_pid, h_mask, in_state, state_num, dim_size):
        """
        历史行为和全局状态联合编码器
        """
        # [batch, state_num, dim_size]
        in_state = self.state_encoder(in_state, state_num, dim_size, hidden_size=100)
        in_state = tf.reshape(in_state, [-1, state_num, dim_size])

        p_size = 3
        max_position = 54
        p_type_table = tf.get_variable(name='p_tag', shape=[p_size, self.dim_size], initializer=create_initializer())
        one_hot_ids = tf.one_hot(tf.reshape(h_pid, [-1]), depth=p_size)
        p_type_embed = tf.reshape(tf.matmul(one_hot_ids, p_type_table), get_shape_list(h_vec))
        h_vec += p_type_embed
        batch_size, h_len = get_shape_list(h_vec)[0], get_shape_list(h_vec)[1]
        assert_op = tf.assert_less_equal(h_len, max_position)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(name='position_embedding', shape=[max_position, self.dim_size],
                                                       initializer=create_initializer())
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [h_len, -1])
            num_dims = len(h_vec.shape.as_list())

            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([h_len, self.dim_size])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            h_vec += position_embeddings
        h_mask = tf.concat([tf.ones([batch_size, state_num]), h_mask], axis=1)
        in_state = tf.concat([in_state, h_vec], axis=1)
        for i in range(3):
            with tf.variable_scope('state_transformer_' + str(i)):
                in_state = vanilla_transformer(in_state, in_state, use_sigmoid=(i % 2 == 0), attention_mask=h_mask,
                                               num_hidden_layers=1)
        return in_state[:, :state_num, :]

    def build_a(self, in_tensor, in_state, in_a_mask, in_l_mask, vocab_size, in_ids, h_vec, h_pid, h_m):
        """
        搭建斗地主的actor模型
        """

        state_num = self.state_num
        orig_shape = get_shape_list(in_tensor)
        batch_size, combo_num, combo_length, dim_size = orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]

        # 动作是可选择的或不可选择的，提前告知模型
        type_embeddings = self.embedding_postprocessor(in_l_mask, batch_size, combo_num, combo_length, dim_size)
        in_tensor = in_tensor + type_embeddings

        # 编码state和对战历史
        # [batch, state_num, dim_size]
        in_state = self.hns_encoder(h_vec, h_pid, h_m, in_state, state_num, dim_size)
        """
        step 1,2,3
        """
        middle_output, outer_output, outer_mask = self.interaction(in_tensor, in_state, in_a_mask, combo_num,
                                                                   combo_length, dim_size, state_num)
        choose_prob = self.choose_combo_set(outer_output, outer_mask)
        outer_output = tf.reshape(outer_output, [-1, combo_num, 1, state_num * dim_size])
        outer_output = tf.tile(outer_output, [1, 1, combo_length, 1])

        output = tf.concat([outer_output, middle_output], axis=3)
        # [batch, combo_num, state_num * dim_size]
        """
        获得动作集内每个动作的出牌率占比
        """
        intermediate_output = tf.layers.dense(output, state_num * dim_size * 4, activation=gelu)
        intermediate_output = tf.layers.dense(intermediate_output, state_num * dim_size * 2)
        output = layer_norm(intermediate_output + output)
        logit = tf.layers.dense(output, 1)
        # [batch, combo_num, combo_length]
        logit = tf.reshape(logit, [batch_size, combo_num, combo_length]) - tf.cast(in_l_mask, tf.float32) * 9999.0
        logit = tf.reshape(logit, [-1, combo_num * combo_length])
        attn = tf.nn.softmax(logit, axis=-1)

        """
        使用选择概率对概率分布进行加权，得到每一动作的实际出牌概率
        """
        # [batch, combo_num, combo_length, vocab_size]
        inputs_onehot = tf.one_hot(tf.reshape(in_ids, [-1, combo_num * combo_length]), depth=vocab_size,
                                   dtype=tf.float32)
        # [b, vocab_size]
        attn_dists = tf.einsum('bf,bfv->bv', attn, inputs_onehot)

        return attn_dists

    def build_c(self, in_tensor, in_state, in_a_mask, in_l_mask, h_vec, h_pid, h_m):
        """
        搭建斗地主的critic模型
        """
        state_num = self.state_num
        orig_shape = get_shape_list(in_tensor)
        batch_size, combo_num, combo_length, dim_size = orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]
        # 动作是可选择的或不可选择的，提前告知模型
        type_embeddings = self.embedding_postprocessor(in_l_mask, batch_size, combo_num, combo_length, dim_size)
        in_tensor = in_tensor + type_embeddings

        # [batch, state_num, dim_size]
        in_state = self.hns_encoder(h_vec, h_pid, h_m, in_state, state_num, dim_size)
        """
        step 1,2,3
        """
        _, outer_output, outer_mask = self.interaction(in_tensor, in_state, in_a_mask,
                                                       combo_num, combo_length, dim_size, state_num)
        choose_prob = self.choose_combo_set(outer_output, outer_mask)
        q = self.reflection(outer_output, in_state, combo_num, dim_size, state_num, act=tf.nn.tanh, out_size=1) * 3

        # q = self.reflection(outer_output, in_state, combo_num, dim_size, state_num, out_size=1)

        q_prob = q * choose_prob
        prob = tf.reduce_sum(q_prob, axis=1)
        return prob

    def build_e(self, in_tensor, in_position, in_a_mask):
        """
        搭建评估器模型
        """
        state_num = self.state_num
        orig_shape = get_shape_list(in_tensor)
        batch_size, combo_num, combo_length, dim_size = orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]
        type_embeddings = self.embedding_postprocessor(in_position, batch_size, combo_num, combo_length, dim_size,
                                                       type_size=3)
        in_tensor = in_tensor + type_embeddings
        attn_mask = tf.reshape(in_a_mask, [-1, combo_length])
        m_attn_mask = tf.concat([attn_mask, tf.ones([batch_size * combo_num, 1])], axis=-1)
        """
        1.动作集内部交互
        """
        state_tag = tf.get_variable(name='state_tag', shape=[1, 1, state_num, self.dim_size],
                                    initializer=create_initializer())
        state_tag = tf.tile(state_tag, [batch_size * combo_num, 1, 1, 1])

        output = []
        for j in range(state_num):
            with tf.variable_scope('middle_transformer' + str(j)):
                in_tensor = tf.reshape(in_tensor, [-1, combo_length, dim_size])
                inner_output = tf.concat([in_tensor, state_tag[:, :, j, :]], axis=1)
                with tf.variable_scope('inner_transformer'):
                    # [batch*3*combo_num, combo_length, dim_size]
                    if j % 2 == 0:
                        sigmoid_out = vanilla_transformer(inner_output, inner_output, use_sigmoid=True,
                                                          attention_mask=m_attn_mask, num_hidden_layers=3)
                    else:
                        sigmoid_out = vanilla_transformer(inner_output, inner_output, use_sigmoid=False,
                                                          attention_mask=m_attn_mask, num_hidden_layers=3)
                    output.append(sigmoid_out[:, -1, :])
        # [batch*3*combo_num, dim_size*state_num]
        combo_state = tf.concat(output, axis=-1)
        hidden_size = self.dim_size * 3
        intermediate_output = tf.layers.dense(combo_state, dim_size * state_num, activation=gelu,
                                              kernel_initializer=create_initializer())
        combo_state = tf.layers.dense(intermediate_output, hidden_size, kernel_initializer=create_initializer())
        """
        3.不同动作集通过独立状态进行交互，减弱重合增益现象
        """
        outer_mask = tf.reshape(attn_mask, [-1, 3 * combo_num, combo_length])[:, :, 0]
        with tf.variable_scope('outer_transformer'):
            # [batch, 3*combo_num, dim_size*state_num]
            combo_state = tf.reshape(combo_state, [-1, 3 * combo_num, hidden_size])
            # [batch, 3*combo_num, state_num * dim_size]
            outer_output = self_transformer(combo_state, attention_mask=outer_mask, num_hidden_layers=3,
                                            hidden_size=hidden_size, intermediate_size=hidden_size,
                                            use_sigmoid=True)

        # [batch, 3*combo_num, 1]
        outer_mask = tf.expand_dims(outer_mask, axis=2)
        choose_logit = tf.layers.dense(outer_output, 1)
        choose_logit = choose_logit - (1.0 - tf.cast(outer_mask, tf.float32)) * 9999.0
        choose_prob = tf.nn.softmax(choose_logit, axis=1)
        q = tf.layers.dense(outer_output, 2, activation=tf.nn.softmax) * choose_prob
        prob = tf.reduce_sum(q, axis=1)
        return prob

    def build_y(self, in_tensor, in_state, in_a_mask):
        """
        搭建抢地主模型
        """
        state_num = self.state_num
        orig_shape = get_shape_list(in_tensor)
        batch_size, combo_num, combo_length, dim_size = orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]

        # [batch, state_num, dim_size]
        in_state = self.state_encoder(in_state, state_num, dim_size, hidden_size=100)
        """
        step 1,2,3
        """
        _, outer_output, outer_mask = self.interaction(in_tensor, in_state, in_a_mask,
                                                       combo_num, combo_length, dim_size, state_num)
        choose_l_prob = self.choose_combo_set(outer_output, outer_mask)
        q_l = self.reflection(outer_output, in_state, combo_num, dim_size, state_num, out_size=2)
        q_l_prob = tf.nn.softmax(q_l, axis=-1) * choose_l_prob
        l_prob = tf.reduce_sum(q_l_prob, axis=1)
        with tf.variable_scope('f'):
            choose_f_prob = self.choose_combo_set(outer_output, outer_mask)
            q_f = self.reflection(outer_output, in_state, combo_num, dim_size, state_num, out_size=2)
            q_f_prob = tf.nn.softmax(q_f, axis=-1) * choose_f_prob
            f_prob = tf.reduce_sum(q_f_prob, axis=1)

        return l_prob, f_prob

    @staticmethod
    def state_encoder(in_state, state_num, dim_size, hidden_size=100):
        """
        状态编码器
        """
        state = tf.layers.dense(in_state, hidden_size)
        state = tf.layers.dense(state, hidden_size, activation=gelu)
        state = tf.layers.dense(state, state_num * dim_size)

        return state

    @staticmethod
    def interaction(in_tensor, in_state, in_a_mask, combo_num, combo_length, dim_size, state_num):
        """
        牌集内外交互

        """
        attn_mask = tf.reshape(in_a_mask, [-1, combo_length])
        """
        1.动作集内部交互
        """
        with tf.variable_scope('inner_transformer'):
            inner_output = tf.reshape(in_tensor, [-1, combo_length, dim_size])
            # [batch*combo_num, combo_length, dim_size]
            inner_output = self_transformer(inner_output, attention_mask=attn_mask, num_hidden_layers=2)
        """
        2.引入外部信息，更新成双向通信后的state和act
        为了引用非同质化的state，采用多个独立transformer进行交互
        """
        # [batch*combo_num, state_num, dim_size]
        state = tf.reshape(in_state, [-1, 1, state_num, dim_size])
        state = tf.tile(state, [1, combo_num, 1, 1])
        state = tf.reshape(state, [-1, state_num, dim_size])
        middle_input = tf.reshape(inner_output, [-1, combo_length, 1, dim_size])
        middle_input = tf.tile(middle_input, [1, 1, state_num, 1])
        for i in range(3):
            states = []
            middles = []
            for j in range(state_num):
                with tf.variable_scope('middle_transformer_layer' + str(i) + '_num_' + str(j)):
                    middle_info = vanilla_transformer(middle_input[:, :, j, :], state[:, j:j + 1, :],
                                                      num_hidden_layers=1)
                    middle_info = tf.reshape(middle_info, [-1, combo_length, 1, dim_size])
                    # state信息注入
                    middles.append(middle_info)
            middle_input = tf.concat(middles, axis=2)
            for j in range(state_num):
                if j % 2 == 0:
                    use_sigmoid = True
                else:
                    use_sigmoid = False
                # use_sigmoid = False
                with tf.variable_scope('state_transformer_layer' + str(i) + '_num_' + str(j)):
                    # state感知牌集
                    states.append(
                        vanilla_transformer(state[:, j:j + 1, :], middle_input[:, :, j, :], use_sigmoid=use_sigmoid,
                                            attention_mask=attn_mask, num_hidden_layers=1))
            state = tf.concat(states, axis=1)

        middle_output = tf.reshape(middle_input, [-1, combo_num, combo_length, state_num * dim_size])
        combo_state = state
        """
        3.不同动作集通过独立状态进行交互，减弱重合增益现象
        """
        outer_mask = tf.reshape(attn_mask, [-1, combo_num, combo_length])[:, :, 0]
        with tf.variable_scope('outer_transformer'):
            outer_vec = tf.reshape(combo_state, [-1, combo_num, state_num * dim_size])
            # [batch, combo_num, state_num * dim_size]
            outer_output = self_transformer(outer_vec, attention_mask=outer_mask, num_hidden_layers=2,
                                            hidden_size=state_num * dim_size, intermediate_size=state_num * dim_size,
                                            use_sigmoid=True)

        return middle_output, outer_output, outer_mask

    @staticmethod
    def choose_combo_set(outer_output, outer_mask):
        """
        获得每个牌组集合的被采用概率
        """
        # [batch, combo_num, 1]
        outer_mask = tf.expand_dims(outer_mask, axis=2)
        choose_logit = tf.layers.dense(outer_output, 1)
        choose_logit = choose_logit - (1.0 - tf.cast(outer_mask, tf.float32)) * 9999.0
        choose_prob = tf.nn.softmax(choose_logit, axis=1)
        return choose_prob

    @staticmethod
    def reflection(outer_output, in_state, combo_num, dim_size, state_num, out_size, act=None):
        """
        整体局势推理
        """
        # b, d*s
        q_state = tf.reshape(in_state, [-1, 1, dim_size * state_num])
        q_state = tf.layers.dense(q_state, dim_size * state_num)
        q_state = tf.tile(q_state, [1, combo_num, 1])
        q_outer = tf.concat([outer_output, q_state], axis=-1)

        intermediate_output = tf.layers.dense(q_outer, state_num * dim_size * 4, activation=gelu)
        intermediate_output = tf.layers.dense(intermediate_output, state_num * dim_size * 2)
        q_outer = layer_norm(intermediate_output + q_outer)

        q = tf.layers.dense(q_outer, out_size, activation=act)
        return q

    def e_act(self, e_act_ids, e_dyn_vec, e_label_mask, e_attn_mask, test_mode=False):
        """
        游戏采样策略
        """
        next_epi = False
        act_ids, label_masks, attn_masks, dyn_vec = convert_alad(e_act_ids, e_label_mask, e_attn_mask,
                                                                 [e_dyn_vec])
        act_nd_ids = [[[[int(idx / 3), k] for k in j] for j in i] for idx, i in enumerate(act_ids)]
        a_m = np.array(attn_masks)
        l_m = np.array(label_masks)
        d_v = np.array(dyn_vec)
        act_nd_ids = np.array(act_nd_ids)
        e = self.sess.run(self.e, {self.act_nd_ids: act_nd_ids, self.l_m: l_m, self.a_m: a_m, self.d_v: d_v})
        if test_mode:
            print('---评估器认为地主胜率为：', e[0][1])
        act_type = np.argmax(e[0])
        rate = 2 * np.min(e[0]) - 0.05
        # 一定概率继续游戏
        rd = random.random()
        if rd > rate and self.e_pointer > 80000:
            next_epi = True

        return act_type, next_epi

    def y_act(self, state, act_ids, attn_masks, dyn_vec, win_first=True, test_mode=False, is_training=False):
        """
        抢地主行为策略
        """
        # 一定概率强制抢地主
        rd = random.random()
        if is_training:
            if rd < 0.2:
                return 1
            elif rd < 0.5:
                return 0

        act_ids, attn_masks, dyn_vec = convert_aad([act_ids], [attn_masks], [dyn_vec])
        act_nd_ids = [[[[idx, k] for k in j] for j in i] for idx, i in enumerate(act_ids)]
        s = np.array([state])
        a_m = np.array(attn_masks)
        d_v = np.array(dyn_vec)
        act_nd_ids = np.array(act_nd_ids)
        y_l, y_f = self.sess.run([self.y_l, self.y_f],
                                 {self.act_nd_ids: act_nd_ids, self.S: s, self.a_m: a_m, self.d_v: d_v})
        if test_mode:
            print('当地主胜率', y_l[0][1], '当农民胜率', y_f[0][1])
        # 寻找最大胜率的方案
        if win_first:
            act_type = np.argmax([y_f[0][1], y_l[0][1]])
        # 寻找奖励期望值最大的方案（地主的奖惩翻倍）
        else:
            act_type = np.argmax([(y_f[0][1] - y_f[0][0]), (y_l[0][1] - y_l[0][0]) * 2])
        return act_type

    def f_act(self, state, act_ids, label_masks, attn_masks, dyn_vec, h_vec, h_pid):
        """
        斗地主行为策略
        """
        act_ids, label_masks, attn_masks, dyn_vec = convert_alad([act_ids], [label_masks], [attn_masks], [dyn_vec])
        act_nd_ids = [[[[idx, k] for k in j] for j in i] for idx, i in enumerate(act_ids)]

        a = self.sess.run(self.a,
                          {self.S: [state], self.a_m: attn_masks, self.act_ids: act_ids,
                           self.act_nd_ids: act_nd_ids, self.h_vec: [h_vec], self.h_pid: [h_pid],
                           self.d_v: dyn_vec, self.l_m: label_masks, self.h_m: [[1 for _ in h_vec]]})
        old_p = np.max(a[0])
        act_type = np.argmax(a[0])

        return act_type, old_p

    def learn_y(self):
        """
        学习抢地主模型
        """
        if self.y_pointer < self.memory_size:
            return [0]
        # 数据采样
        indices = np.random.choice(self.memory_size, size=self.batch_size)
        state, act_ids, attn_masks, dyn_vec, y_l_cls, y_f_cls, loss_mask = [], [], [], [], [], [], []
        for i, idx in enumerate(indices):
            state.append(self.y_memory['s'][idx])
            act_ids.append(self.y_memory['a_s'][idx])
            attn_masks.append(self.y_memory['a_m'][idx])
            dyn_vec.append(self.y_memory['d_v'][idx])
            y_l_cls.append(self.y_memory['y_l_cls'][idx])
            y_f_cls.append(self.y_memory['y_f_cls'][idx])
            loss_mask.append(self.y_memory['loss_mask'][idx])

        act_ids, attn_masks, dyn_vec = convert_aad(act_ids, attn_masks, dyn_vec)
        act_nd_ids = [[[[idx, k] for k in j] for j in i] for idx, i in enumerate(act_ids)]

        _, y_loss = self.sess.run([self.y_train, self.y_loss],
                                  {self.S: state,
                                   self.a_m: attn_masks, self.act_ids: act_ids,
                                   self.act_nd_ids: act_nd_ids, self.loss_mask: loss_mask,
                                   self.y_l_cls: y_l_cls, self.y_f_cls: y_f_cls, self.d_v: dyn_vec})
        return [y_loss]

    def learn_e(self):
        """
        学习评估器模型
        """
        if self.e_pointer < self.memory_size:
            return [0]
        # 数据采样
        indices = np.random.choice(self.memory_size, size=int(self.batch_size / 3))
        act_ids, attn_masks, label_masks, dyn_vec, e_cls = [], [], [], [], []
        for i, idx in enumerate(indices):

            dyn_vec.append(self.e_memory['d_v'][idx])
            e_cls.append(self.e_memory['e_cls'][idx])
            for j in range(3):
                act_ids.append(self.e_memory['a_s'][idx][j])
                attn_masks.append(self.e_memory['a_m'][idx][j])
                label_masks.append(self.e_memory['l_m'][idx][j])

        act_ids, label_masks, attn_masks, dyn_vec = convert_alad(act_ids, label_masks, attn_masks, dyn_vec)
        act_nd_ids = [[[[int(idx / 3), k] for k in j] for j in i] for idx, i in enumerate(act_ids)]

        _, e_loss = self.sess.run([self.e_train, self.e_loss],
                                  {self.a_m: attn_masks, self.l_m: label_masks,
                                   self.act_nd_ids: act_nd_ids,
                                   self.e_cls: e_cls, self.d_v: dyn_vec})
        return [e_loss]

    def learn_f(self):
        """
        学习斗地主模型
        """
        if self.f_pointer < self.memory_size:
            return [0, 0]
        batch_num = int(self.memory_size / self.batch_size)
        num_epochs = 3
        a_losses, c_losses = [], []
        for _ in range(num_epochs):
            indices = np.random.permutation(np.arange(self.memory_size))
            for j in range(batch_num):

                batch_indices = indices[int(j * (self.memory_size / batch_num)): int((j + 1) * (
                        self.memory_size / batch_num))]

                state, act_ids, label_masks, attn_masks, dyn_vec, action, reward = [], [], [], [], [], [], []

                h_vec, h_pid = [], []
                old_p = []

                for i, idx in enumerate(batch_indices):
                    h_vec.append(self.f_memory['h_vec'][idx])
                    h_pid.append(self.f_memory['h_pid'][idx])
                    state.append(self.f_memory['s'][idx])
                    act_ids.append(self.f_memory['a_s'][idx])
                    label_masks.append(self.f_memory['l_m'][idx])
                    attn_masks.append(self.f_memory['a_m'][idx])
                    dyn_vec.append(self.f_memory['d_v'][idx])
                    action.append([i, self.f_memory['a'][idx]])
                    reward.append(self.f_memory['r'][idx])
                    old_p.append(self.f_memory['old_p'][idx])
                act_ids, label_masks, attn_masks, dyn_vec = convert_alad(act_ids, label_masks, attn_masks, dyn_vec)

                h_vec, h_pid, h_m = convert_h(h_vec, h_pid)
                act_nd_ids = [[[[idx, k] for k in j] for j in i] for idx, i in enumerate(act_ids)]

                _, _, a_loss, c_loss = self.sess.run([self.a_train, self.c_train, self.a_loss, self.c_loss],
                                                     {self.S: state, self.l_m: label_masks,
                                                      self.a_m: attn_masks, self.h_vec: h_vec, self.h_pid: h_pid,
                                                      self.act_ids: act_ids, self.h_m: h_m,
                                                      self.act_nd_ids: act_nd_ids,
                                                      self.R: reward, self.d_v: dyn_vec,
                                                      self.f_action: action, self.old_p: old_p})
                a_losses.append(a_loss)
                c_losses.append(c_loss)
        self.f_pointer = 0
        a_loss = np.mean(a_losses)
        c_loss = np.mean(c_losses)

        return [a_loss, c_loss]

    def store_f(self, _state, _act_ids, _label_masks, _attn_masks, _dyn_vec, _action, _h_vec, _h_pid, reward,
                 old_p):
        """
        存储斗地主记忆
        """
        if 's' not in self.f_memory:
            for name in ['s', 'a_s', 'd_v', 'l_m', 'a_m', 'h_pid', 'h_vec', 'a', 'r', 'old_p']:
                self.f_memory[name] = [[] for _ in range(self.memory_size)]
        index = self.f_pointer % self.memory_size
        self.f_memory['s'][index] = _state
        self.f_memory['a_s'][index] = _act_ids
        self.f_memory['d_v'][index] = _dyn_vec
        self.f_memory['l_m'][index] = _label_masks
        self.f_memory['a_m'][index] = _attn_masks
        self.f_memory['h_vec'][index] = _h_vec
        self.f_memory['h_pid'][index] = _h_pid

        self.f_memory['a'][index] = _action
        self.f_memory['r'][index] = [reward]

        self.f_memory['old_p'][index] = old_p
        self.f_pointer += 1

    def store_y(self, _state, _act_ids, attn_masks, _dyn_vec, y_l_cls, y_f_cls, loss_mask):
        """
        存储抢地主记忆
        """
        if 's' not in self.y_memory:
            for name in ['s', 'a_s', 'd_v', 'a_m', 'y_l_cls', 'y_f_cls', 'loss_mask']:
                self.y_memory[name] = [[] for _ in range(self.memory_size)]
        index = self.y_pointer % self.memory_size
        self.y_memory['s'][index] = _state
        self.y_memory['a_s'][index] = _act_ids
        self.y_memory['d_v'][index] = _dyn_vec
        self.y_memory['a_m'][index] = attn_masks
        self.y_memory['y_l_cls'][index] = y_l_cls
        self.y_memory['y_f_cls'][index] = y_f_cls
        self.y_memory['loss_mask'][index] = loss_mask

        self.y_pointer += 1

    def store_e(self, e_act_ids, e_dyn_vec, e_label_mask, e_attn_mask, e_cls):
        """
        存储评估器记忆
        """
        if 'a_s' not in self.e_memory:
            for name in ['a_s', 'd_v', 'l_m', 'a_m', 'e_cls']:
                self.e_memory[name] = [[] for _ in range(self.memory_size)]
        index = self.e_pointer % self.memory_size
        self.e_memory['a_s'][index] = e_act_ids
        self.e_memory['d_v'][index] = e_dyn_vec
        self.e_memory['l_m'][index] = e_label_mask
        self.e_memory['a_m'][index] = e_attn_mask
        self.e_memory['e_cls'][index] = e_cls
        self.e_pointer += 1

    def save_model(self, name, episode):
        # 存储模型
        saver = tf.train.Saver()
        saver.save(self.sess, "data/" + name + "_" + str(episode) + ".ckpt")

    def load_model(self, name, episode):
        # 加载模型
        saver = tf.train.Saver()
        saver.restore(self.sess, "data/" + name + "_" + str(episode) + ".ckpt")
