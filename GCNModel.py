from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import threading
import h5py

import util
from util import *
import time


class KnowledgePronounCorefModel(object):
    def __init__(self, config):
        self.config = config
        self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.softmax_threshold = config['softmax_threshold']
        if config["lm_path"]:
            self.lm_file = h5py.File(self.config["lm_path"], "r")
        else:
            self.lm_file = None
        self.lm_layers = self.config["lm_layers"]  # 3
        self.lm_size = self.config["lm_size"]  # 1024
        self.eval_data = None  # Load eval data lazily.
        print('Start to load the eval data')
        st = time.time()
        if not config["predict"]:
            self.load_eval_data()
        print("Finished in {:.2f}".format(time.time() - st))

        input_props = []
        input_props.append((tf.string, [None, None]))  # Tokens.
        input_props.append((tf.float32, [None, None]))  # adj.
        input_props.append((tf.float32, [None, None]))  # xor.
        input_props.append((tf.float32, [None, None, self.context_embeddings.size]))  # Context embeddings.
        input_props.append((tf.float32, [None, None, self.head_embeddings.size]))  # Head embeddings.
        input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, []))  # pronoun lengths.
        input_props.append((tf.int32, []))  # name lengths.
        input_props.append((tf.int32, []))  # status lengths.
        input_props.append((tf.int32, [None]))  # Speaker IDs.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # gold_starts.
        input_props.append((tf.int32, [None]))  # gold_ends.
        input_props.append((tf.int32, [None, None]))  # number_features.
        input_props.append((tf.int32, [None, None]))  # candidate_positions.
        input_props.append((tf.int32, [None, None]))  # pronoun_positions.
        input_props.append((tf.int32, [None, None]))  # name_position.
        input_props.append((tf.int32, [None, None]))  # status_positions.
        input_props.append((tf.int32, [None, None]))  # order_features.
        input_props.append((tf.bool, [None, None]))  # labels
        input_props.append((tf.float32, [None, None]))  # candidate_masks

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        self.predictions, self.loss, self.summaries = self.get_predictions_and_loss(*self.input_tensors)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                                   self.config["decay_frequency"], self.config["decay_rate"],
                                                   staircase=True)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                for example in train_examples:
                    tensorized_example = self.tensorize_pronoun_example(example, is_training=True)
                    feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                    session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def restore(self, session, log_path=None):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
        saver = tf.train.Saver(vars_to_restore)
        if log_path:
            checkpoint_path = os.path.join(log_path, "model.max.ckpt")
        else:
            checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def load_lm_embeddings(self, doc_key):
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        group = self.lm_file[doc_key]
        num_sentences = len(list(group.keys()))  # number of sentence
        sentences = [group[str(i)][...] for i in range(num_sentences)]

        # 句子个数 * 最大句长
        lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_mention_list(self, gold_mentions):
        lens = []
        for mention in gold_mentions:
            if type(mention[0]) == int:
                lens.append(1)
            else:
                lens.append(len(mention))
        max_len = max(lens)
        starts = np.zeros((len(gold_mentions), max_len))
        ends = np.zeros((len(gold_mentions), max_len))
        starts.fill(-1)
        ends.fill(-1)
        for i in range(len(gold_mentions)):
            mention = gold_mentions[i]
            if type(mention[0]) == int:
                starts[i, 0] = mention[0]
                ends[i, 0] = mention[0]
            else:
                for j in range(len(mention)):
                    tar_mention = mention[j]
                    starts[i, j] = tar_mention[0]
                    ends[i, j] = tar_mention[1]
        return starts, ends

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_pronoun_example(self, example, is_training):
        """
        notice: input_props.append((tf.int32, [None]))  # gold_starts or gold_ends
        :param example: input example
        :param is_training:
        :return: tensorized example
        """
        gold_mentions = list()
        for pronoun_example in example['pronoun_info']:
            gold_mentions.append(pronoun_example['current_pronoun'])
            for tmp_np in pronoun_example['candidate_NPs']:
                if tmp_np not in gold_mentions and tmp_np[1] - tmp_np[0] < self.config["max_span_width"]:
                    gold_mentions.append(tmp_np)

        names_len = 0
        for idx, name in enumerate(example['status_name_info']['status_or_name']):
            names_len += 1
            if name not in gold_mentions and name[1] - name[0] < self.config["max_span_width"]:
                gold_mentions.append(name)

        # gold_mentions = sorted(gold_mentions)

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util.flatten(example["speakers"])

        assert num_words == len(speakers)

        # if not is_training:
        #     save_tokens_list = [x for s in sentences for x in s]
        #     with open('./attention_input.txt', 'a') as f:
        #         json.dump(save_tokens_list, f)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]

        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
        tokens = np.array(tokens)

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"].strip()

        lm_emb = self.load_lm_embeddings(doc_key)

        num_candidate_NPs = list()
        for i, pronoun_example in enumerate(example['pronoun_info']):
            num_candidate_NPs.append(len(pronoun_example['candidate_NPs']))

        max_candidate_NP_length = max(num_candidate_NPs)

        # adj matrix  [P, S, M]
        adj = np.array(example['adj'])
        length = adj.shape[0]
        if self.config["use_xor_matrix"]:
            ones = np.ones([length, length])
            xor = 1 * np.logical_xor(adj, ones)
        else:
            xor = np.zeros([length, length])

        assert adj.shape[0] == len(example['pronoun_info']) + max_candidate_NP_length + names_len

        candidate_NP_positions = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
        name_positions = np.zeros([names_len, 1])  # [c2, 1]
        pronoun_positions = np.zeros([len(example['pronoun_info']), 1])
        status_positions = np.zeros([max_candidate_NP_length, 1])  # [c, 1]
        labels = np.zeros([len(example['pronoun_info']), max_candidate_NP_length], dtype=bool)
        candidate_mask = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])

        for i, name_example in enumerate(example['status_name_info']['status_or_name']):
            for k, tmp_tuple in enumerate(gold_mentions):
                if name_example == tmp_tuple:
                    name_positions[i, 0] = k

        for i, pronoun_example in enumerate(example['pronoun_info']):
            for j, tmp_np in enumerate(pronoun_example['candidate_NPs']):
                candidate_mask[i, j] = 1
                for k, tmp_tuple in enumerate(gold_mentions):
                    if tmp_tuple == tmp_np:
                        candidate_NP_positions[i, j] = k
                        status_positions[j, 0] = k
                        break

                if tmp_np in pronoun_example['correct_NPs']:
                    labels[i, j] = 1

            for m, tmp_tuple in enumerate(gold_mentions):
                if tmp_tuple == pronoun_example['current_pronoun']:
                    pronoun_positions[i, 0] = m
                    break

        p_len = pronoun_positions.shape[0]
        n_len = names_len  # number of names
        s_len = max_candidate_NP_length

        number_features = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
        order_features = np.zeros([len(example['pronoun_info']), 1])  # [k, 1]
        for idx, pronoun_example in enumerate(example['pronoun_info']):
            order_features[idx, 0] = pronoun_example['pronoun_features']['order']

        example_tensors = (tokens, adj, xor, context_word_emb, head_word_emb, lm_emb, text_len, p_len, n_len,
                           s_len, speaker_ids, is_training, gold_starts, gold_ends, number_features, candidate_NP_positions,
                           pronoun_positions, name_positions, status_positions, order_features, labels, candidate_mask)

        return example_tensors

    def is_in_gold_mention(self, tmp_np, gold_mentions):
        if len(tmp_np) == 1:
            if type(tmp_np[0]) == int:
                if tmp_np not in gold_mentions and tmp_np[1] - tmp_np[0] < self.config["max_span_width"]:
                    return False
                return True
            else:
                if tmp_np not in gold_mentions and tmp_np[0][1] - tmp_np[0][0] < self.config["max_span_width"]:
                    return False
                return True
        else:
            for item in tmp_np:
                if item[1] - item[0] >= self.config["max_span_width"]:
                    return True
            for mention in gold_mentions:
                if tmp_np == mention:
                    return True
            return False

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def get_predictions_and_loss(self, tokens, adj, xor, context_word_emb, head_word_emb, lm_emb, text_len, p_len,
                                 m_len, s_len, speaker_ids, is_training, gold_starts, gold_ends, number_features,
                                 candidate_positions, pronoun_positions, name_positions, status_positions,
                                 order_features, labels, candidate_mask):
        all_k = util.shape(number_features, 0)
        all_c = util.shape(number_features, 1)

        #  dropout
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(context_word_emb)[0]  # 当前example的sentence个数
        max_sentence_length = tf.shape(context_word_emb)[1]  # sentences中最长的句子长度

        context_emb_list = [context_word_emb] if self.config['use_word_emb'] else []

        head_emb_list = [head_word_emb]

        lm_emb_size = util.shape(lm_emb, 2)  # 1024
        lm_num_layers = util.shape(lm_emb, 3)  # 3
        with tf.variable_scope("lm_aggregation"):
            self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
            self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))

        # reshape lm_emb [?, 3]
        flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])

        # lm_emb matmul weight matrix [num_sentences * max_sentence_length * emb, 1]
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1))  # [?, 1]

        # lm_emb reshape [?, ?, 1024]
        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
        aggregated_lm_emb *= self.lm_scaling

        # add elmo emb to context_emb_list
        if self.config['use_elmo']:
            context_emb_list.append(aggregated_lm_emb)

        # [num_sentences, max_sentence_length, emb] [?, ?, 1474]
        context_emb = tf.concat(context_emb_list, 2)

        # [num_sentences, max_sentence_length, emb] [?, ?, 450]
        head_emb = tf.concat(head_emb_list, 2)

        # [num_sentences, max_sentence_length, emb] [?, ?, 1474]
        context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)

        # [num_sentences, max_sentence_length, emb] [?, ?, 450]
        head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)

        # [num_sentence, max_sentence_length]
        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)

        # context to lstm [num_words, emb] [?, 400]
        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)

        if self.config["use_word_attn"]:
            context_outputs = tf.expand_dims(context_outputs, 0)
            attn_out = util.attention_layer(context_outputs, context_outputs, is_training,
                                            num_attention_heads=4, size_per_head=100)
            context_outputs = tf.squeeze(context_outputs, 0)
            attn_out = tf.squeeze(attn_out, 0)
            context_outputs = tf.add(context_outputs, attn_out)

        # [num_words] [?, 450]
        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)

        top_span_starts = gold_starts
        top_span_ends = gold_ends

        # get span emb [?, 1270]
        top_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, top_span_starts, top_span_ends)

        # [k, max_candidate, embedding] [?, ?, 1270]
        candidate_NP_embeddings = tf.gather(top_span_emb, candidate_positions)

        # [k3, embedding] [k2, 1, embedding]
        name_embedding = tf.gather(top_span_emb, name_positions)

        # [k1, embedding]  [k1, 1, embedding]
        pronoun_embedding = tf.gather(top_span_emb, pronoun_positions)

        status_embedding = tf.gather(top_span_emb, status_positions)

        # [k, max_candidate]
        if self.config["use_multi_span"]:
            head_span_starts = tf.squeeze(top_span_starts[:, :1], 1)  # todo model_heads
            candidate_starts = tf.gather(head_span_starts, candidate_positions)
            pronoun_starts = tf.gather(head_span_starts, pronoun_positions)  # [k, 1] [?, ?]
        else:
            candidate_starts = tf.gather(top_span_starts, candidate_positions)  # [k, max_candidate]
            pronoun_starts = tf.gather(top_span_starts, pronoun_positions)  # [k, 1]

        # [k] [?, ?]
        top_span_speaker_ids = tf.gather(speaker_ids, candidate_starts)

        # [k, 1] [?, ?]
        pronoun_speaker_id = tf.gather(speaker_ids, pronoun_starts)

        mention_offsets = tf.range(util.shape(top_span_emb, 0)) + 1
        candidate_NP_offsets = tf.gather(mention_offsets, candidate_positions)
        pronoun_offsets = tf.gather(mention_offsets, pronoun_positions)

        k = util.shape(pronoun_positions, 0)
        dummy_scores = tf.zeros([k, 1])  # [k, 1]

        for i in range(self.config["coref_depth"]):
            with tf.variable_scope("coref_layer", reuse=(i > 0)):
                coreference_scores = self.get_coreference_score(candidate_NP_embeddings, pronoun_embedding,
                                                                top_span_speaker_ids, pronoun_speaker_id,
                                                                candidate_NP_offsets, pronoun_offsets,
                                                                number_features, order_features)
        tf.summary.histogram("coreference_scores", coreference_scores)

        score_after_softmax = tf.nn.softmax(coreference_scores, 1)  # [k, c]
        tf.summary.histogram("coreference_scores_softmax", score_after_softmax)

        if self.config['softmax_pruning']:
            threshold = tf.ones([all_k, all_c]) * self.config['softmax_threshold']  # [k, c]
        else:
            threshold = tf.zeros([all_k, all_c]) - tf.ones([all_k, all_c])
        ranking_mask = tf.to_float(tf.greater(score_after_softmax, threshold))  # [k, c]

        if self.config["apply_biaffine"]:
            with tf.variable_scope("biaffine_layer"):
                emb_list = [pronoun_embedding, status_embedding, name_embedding]
                if self.config["use_PMS_attention"]:
                    attn_out = self.use_attention(emb_list)  # [1, ?, 256]
                    biaffine_score = self.use_attn_biaffine(attn_out, s_len, p_len)

                elif self.config["use_PMS_gcn"]:
                    gcn_out = self.use_gcn(emb_list, adj)  # [1, ?, 128]
                    biaffine_score = self.use_gcn_biaffine(gcn_out, status_embedding, pronoun_embedding, s_len, p_len)
                    tf.summary.histogram("bi_affine_score", biaffine_score)
                else:
                    biaffine_score = tf.zeros([p_len, s_len], dtype=tf.float32)

                coreference_scores = self.config["biaffine_score_weight"] * biaffine_score + \
                    (1 - self.config["biaffine_score_weight"]) * coreference_scores

                # coreference_scores = tf.add(coreference_scores, biaffine_score)

                if self.config["use_xor_matrix"]:
                    xor_mat = tf.slice(xor, begin=[0, p_len], size=[p_len, s_len])  # [p, s]
                    coreference_scores = tf.multiply(coreference_scores, xor_mat)

                if self.config['softmax_biaffine_pruning']:
                    biaffine_score = tf.math.sigmoid(biaffine_score)
                    tf.summary.histogram("bi_affine_score_sigmoid", biaffine_score)

                    biaffine_score_softmax = tf.nn.softmax(biaffine_score, 1)
                    tf.summary.histogram("bi_affine_score_softmax", biaffine_score_softmax)

                    biaffine_threshold = tf.ones([all_k, all_c]) * self.config['softmax_biaffine_threshold']
                    biaffine_ranking_mask = tf.to_float(tf.greater(biaffine_score_softmax, biaffine_threshold))
                    ranking_mask = ranking_mask * biaffine_ranking_mask

        top_antecedent_scores = tf.concat([dummy_scores, coreference_scores], 1)  # [k, c + 1]
        labels = tf.logical_and(labels, tf.greater(score_after_softmax, threshold))

        dummy_mask_1 = tf.ones([k, 1])
        dummy_mask_0 = tf.zeros([k, 1])

        mask_for_prediction = tf.concat([dummy_mask_0, candidate_mask], 1)
        ranking_mask_for_prediction = tf.concat([dummy_mask_0, ranking_mask], 1)

        if self.config['random_sample_training']:
            random_mask = tf.greater(tf.random_uniform([all_k, all_c]), tf.ones([all_k, all_c]) * 0.3)
            labels = tf.logical_and(labels, random_mask)
            ranking_mask = ranking_mask * tf.to_float(random_mask)

        dummy_labels = tf.logical_not(tf.reduce_any(labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, labels], 1)  # [k, c + 1]
        mask_for_training = tf.concat([dummy_mask_1, candidate_mask], 1)
        ranking_mask_for_training = tf.concat([dummy_mask_1, ranking_mask], 1)

        loss = self.softmax_loss(top_antecedent_scores * mask_for_training * ranking_mask_for_training,
                                 top_antecedent_labels)
        loss = tf.reduce_sum(loss)  # []

        summaries = tf.summary.merge_all()

        return [top_antecedent_scores * mask_for_prediction * ranking_mask_for_prediction,
                score_after_softmax * candidate_mask], loss, summaries

    def get_n_s_emb(self, name_map_emb, status_emb):
        name_map_emb = tf.squeeze(name_map_emb, 1)
        status_emb = status_emb[:1, :, :]  # [1, p, 1270]
        status_emb = tf.squeeze(status_emb, 0)
        flatten_emb = tf.concat([status_emb, name_map_emb], 1)  # 2540

        hidden_size = 820
        hidden_weights = tf.get_variable("ns_hidden_weights", [shape(flatten_emb, 1), hidden_size])
        hidden_bias = tf.get_variable("ns_hidden_bias", [hidden_size])
        ns_emb_output = tf.nn.relu(tf.nn.xw_plus_b(flatten_emb, hidden_weights, hidden_bias))
        return ns_emb_output

    def use_gcn(self, emb_list, adj_mat):
        pronoun_embedding, status_embedding, name_embedding = emb_list
        pronoun_embedding = tf.squeeze(pronoun_embedding, 1)  # [P, emb]
        status_embedding = tf.squeeze(status_embedding, 1)  # [S, emb]
        name_embedding = tf.squeeze(name_embedding, 1)  # [M, emb]
        flattened_emb = tf.concat([pronoun_embedding, status_embedding, name_embedding], 0)  # [P+S+M, emb]
        flattened_emb = tf.expand_dims(flattened_emb, 0)  # [1, P+S+M, emb]

        # [1, ?, 128]
        # attn_out = util.attention_layer(flattened_emb, flattened_emb,
        # size_per_head=self.config["size_per_head_with_gcn"])
        # add_out = tf.add(flattened_emb, attn_out)

        gcn_out = util.gcn_layer(flattened_emb, adj_mat)
        return gcn_out

    def use_gcn_biaffine(self, gcn_out, status_embedding, pronoun_embedding, s_len, p_len):
        gcned_pronoun_emb = gcn_out[:, :p_len, :]  # [1, p, 128]
        gcned_status_emb = gcn_out[:, p_len:p_len+s_len, :]  # [1, s, 128]

        pronoun_embedding = tf.transpose(pronoun_embedding, [1, 0, 2])  # [1, p, 820]
        status_embedding = tf.transpose(status_embedding, [1, 0, 2])

        pronoun_embedding = tf.concat([pronoun_embedding, gcned_pronoun_emb], 2)  # [[1, s, 948]
        status_embedding = tf.concat([status_embedding, gcned_status_emb], 2)

        mat = tf.concat([pronoun_embedding, status_embedding], 1)

        biaffine = util.biaffine_layer(mat, mat, 948, self.config["biaffine_out_features"])  # [1, k+s, k+s, 2]
        biaffine = tf.squeeze(biaffine, 0)

        biaffine_pos = tf.slice(biaffine, begin=[0, 0, 1], size=[-1, -1, 1])  # [k+c, k+c, 1]
        biaffine = tf.squeeze(biaffine_pos, 2)

        p_s_mat = tf.slice(biaffine, begin=[0, p_len], size=[p_len, s_len])  # [k, s]
        # s_p_mat = tf.slice(biaffine, begin=[p_len, 0], size=[s_len, p_len])  # [s, k]
        # mat_score = s_p_mat + tf.transpose(p_s_mat, [1, 0])
        return p_s_mat

    def use_attention(self, emb_list):
        # flattened_pronoun_emb, flattened_name_emb = self.emb2cnn(emb_list)
        pronoun_embedding, status_embedding, name_embedding = emb_list
        pronoun_embedding = tf.squeeze(pronoun_embedding, 1)  # [k1, emb]
        status_embedding = tf.squeeze(status_embedding, 1)  # [k3, emb]
        name_embedding = tf.squeeze(name_embedding, 1)  # [k2, emb]

        flattened_emb = tf.concat([pronoun_embedding, status_embedding, name_embedding], 0)  # [k1+k2+k3, emb]
        flattened_emb = tf.expand_dims(flattened_emb, 0)  # [1, k1+k2, emb]

        # [1, ?, 256]
        attn_out = util.attention_layer(flattened_emb, flattened_emb, size_per_head=self.config["size_per_head"])
        return attn_out

    def use_attn_biaffine(self, attn_out, s_len, p_len):
        p_s_out = attn_out[:, :p_len+s_len, :]
        biaffine = util.biaffine_layer(p_s_out, p_s_out, self.config["size_per_head"],
                                       self.config["biaffine_out_features"])  # [1, k+c, k+c]
        biaffine = tf.squeeze(biaffine, 0)  # [k+c, k+c]

        biaffine_pos = tf.slice(biaffine, begin=[0, 0, 1], size=[-1, -1, 1])  # [k+c, k+c, 1]
        biaffine_pos = tf.squeeze(biaffine_pos, 2)  # [k+c, k+c]

        p_s_mat = tf.slice(biaffine_pos, begin=[0, p_len], size=[p_len, s_len])  # [k, c]
        s_p_mat = tf.slice(biaffine_pos, begin=[p_len, 0], size=[s_len, p_len])  # [c, k]

        mat_score = p_s_mat + tf.transpose(s_p_mat, [1, 0])
        return mat_score

    def emb2cnn(self, emb_list):
        pronoun_embedding, name_embedding, status_embedding = emb_list

        pronoun_embedding = tf.transpose(pronoun_embedding, [1, 0, 2])  # 1, k ,emb
        name_embedding = tf.transpose(name_embedding, [1, 0, 2])
        # pronoun_embedding = tf.squeeze(pronoun_embedding, 1)
        # name_embedding = tf.squeeze(name_embedding, 1)

        flattened_pronoun_emb = util.cnn(pronoun_embedding, self.config["emb_filter_widths"],
                                         self.config["emb_filter_size"], name='p_')
        flattened_name_emb = util.cnn(name_embedding, self.config["emb_filter_widths"],
                                      self.config["emb_filter_size"], name='n_')

        return flattened_pronoun_emb, flattened_name_emb

    def get_knowledge_score(self, candidate_NP_embeddings, number_features, candidate_mask):
        k = util.shape(number_features, 0)
        c = util.shape(number_features, 1)

        column_mask = tf.tile(tf.expand_dims(candidate_mask, 1), [1, c, 1])  # [k, c, c]
        row_mask = tf.tile(tf.expand_dims(candidate_mask, 2), [1, 1, c])  # [k, c, c]
        square_mask = column_mask * row_mask  # [k, c, c]

        diagonal_mask = tf.ones([k, c, c]) - tf.tile(tf.expand_dims(tf.diag(tf.ones([c])), 0), [k, 1, 1])
        # we need to find the embedding for these features
        number_emb = tf.gather(tf.get_variable("number_emb", [2, self.config["feature_size"]]),
                               number_features)  # [k, c, feature_size]
        if self.config['number']:
            number_score = self.get_feature_score(number_emb, 'number_score')  # [k, c, c, 1]
        else:
            number_score = tf.zeros([k, c, c, 1])

        merged_score = number_score  # [k, c, c, 1]

        if self.config['attention']:
            if self.config['number']:
                number_attention_score = self.get_feature_attention_score(number_emb, candidate_NP_embeddings,
                                                                          'number_attention_score')
            else:
                number_attention_score = tf.ones([k, c, c, 1]) * -1000

            merged_attention_score = number_attention_score
            all_attention_scores = tf.nn.softmax(merged_attention_score, 3)  # [k, c, c, 2]
            all_scores = merged_score * all_attention_scores
        else:
            all_scores = merged_score
            all_attention_scores = tf.zeros([k, c, c, 4])
        all_scores = tf.reduce_sum(all_scores, 3)  # [k, c, c]
        all_scores = all_scores * diagonal_mask
        all_scores = all_scores * square_mask
        final_score = tf.reduce_mean(all_scores, 2)  # [k, c]

        return final_score, merged_score, all_attention_scores, diagonal_mask, square_mask

    def get_feature_attention_score(self, tmp_feature_emb, tmp_candidate_embedding, tmp_name):
        k = util.shape(tmp_feature_emb, 0)  # [k, c,
        c = util.shape(tmp_feature_emb, 1)
        tmp_feature_size = util.shape(tmp_feature_emb, 2)
        tmp_emb_size = util.shape(tmp_candidate_embedding, 2)
        overall_emb = tf.concat([tmp_candidate_embedding, tmp_feature_emb], 2)  # [k, c, feature_size+embedding_size]

        repeated_emb = tf.tile(tf.expand_dims(overall_emb, 1), [1, c, 1, 1])  # [k, c, c, feature_size+embedding_size]
        tiled_emb = tf.tile(tf.expand_dims(overall_emb, 2), [1, 1, c, 1])  # [k, c, c, feature_size+embedding_size]

        final_feature = tf.concat([repeated_emb, tiled_emb, repeated_emb * tiled_emb],
                                  3)  # [k, c, c, (feature_size+embedding_size)*3]
        final_feature = tf.reshape(final_feature, [k, c * c, (tmp_feature_size + tmp_emb_size) * 3])
        with tf.variable_scope(tmp_name):
            feature_attention_scores = util.ffnn(final_feature, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                                 self.dropout)  # [k, c*c, 1]
        feature_attention_scores = tf.reshape(feature_attention_scores, [k, c, c, 1])
        return feature_attention_scores

    def get_feature_score(self, tmp_feature_emb, tmp_feature_name):
        k = util.shape(tmp_feature_emb, 0)
        c = util.shape(tmp_feature_emb, 1)
        repeated_feature_emb = tf.tile(tf.expand_dims(tmp_feature_emb, 1), [1, c, 1, 1])  # [k, c, c, feature_size]
        tiled_feature_emb = tf.tile(tf.expand_dims(tmp_feature_emb, 2), [1, 1, c, 1])  # [k, c, c, feature_size]

        final_feature = tf.concat([repeated_feature_emb, tiled_feature_emb, repeated_feature_emb * tiled_feature_emb],
                                  3)  # [k, c, c, feature_size*3]
        final_feature = tf.reshape(final_feature,
                                   [k, c * c, self.config["feature_size"] * 3])  # [k, c*c, feature_size*3]

        with tf.variable_scope(tmp_feature_name):
            tmp_feature_scores = util.ffnn(final_feature, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                           self.dropout)  # [k, c*c, 1]
            tmp_feature_scores = tf.reshape(tmp_feature_scores, [k, c, c, 1])  # [k, c, c]
        return tmp_feature_scores

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        if self.config["use_multi_span"]:
            emb_size = 2 * self.config["contextualization_size"]
            dim0 = util.shape(span_starts, 0)
            dim1 = util.shape(span_starts, 1)

            reshaped_span_starts = tf.reshape(span_starts, [dim0 * dim1])  # [a*b]
            gathered_span_starts = tf.gather(context_outputs, reshaped_span_starts)  # [a*b, emb]
            cnn_span_starts = tf.reshape(gathered_span_starts, [dim0, dim1, emb_size])  # [a, b, emb]

            span_starts_4dim = tf.expand_dims(cnn_span_starts, 3)
            span_start_emb = util.cnn2d(span_starts_4dim, self.config["emb_filter_widths"], ffnn_out_size=emb_size, name="start")

            reshaped_span_ends = tf.reshape(span_ends, [dim0 * dim1])
            gathered_span_ends = tf.gather(context_outputs, reshaped_span_ends)
            cnn_span_ends = tf.reshape(gathered_span_ends, [dim0, dim1, emb_size])

            span_ends_4dim = tf.expand_dims(cnn_span_ends, 3)
            span_end_emb = util.cnn2d(span_ends_4dim, self.config["emb_filter_widths"], ffnn_out_size=emb_size, name="end")

            span_starts = tf.squeeze(span_starts[:, :1], 1)  # todo model_heads
            span_ends = tf.squeeze(span_ends[:, :1], 1)  # todo  model_heads

        else:
            span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
            span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]

        span_width = 1 + span_ends - span_starts  # [k]
        span_emb_list.append(span_start_emb)
        span_emb_list.append(span_end_emb)

        if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),
                span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            # [k, max_span_width]
            span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1)
            span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
            span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
            with tf.variable_scope("head_scores"):
                self.head_scores = util.projection(context_outputs, 1)  # [num_words, 1]
            span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
            span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32),
                                       2)  # [k, max_span_width, 1]
            span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
            span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
            span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        return span_emb  # [k, emb]

    def get_mention_scores(self, span_emb):
        with tf.variable_scope("mention_scores"):
            return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)  # [k, 1]

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def bucket_SP_score(self, sp_scores):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(sp_scores)) / math.log(2))) + 3
        use_identity = tf.to_int32(sp_scores <= 4)
        combined_idx = use_identity * sp_scores + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_coreference_score(self, candidate_NPs_emb, pronoun_emb, candidate_NPs_speaker_ids,
                              pronoun_speaker_id, candidate_NP_offsets, pronoun_offsets, number_features, order_features):
        k = util.shape(candidate_NPs_emb, 0)
        c = util.shape(candidate_NPs_emb, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            same_speaker = tf.equal(candidate_NPs_speaker_ids, tf.tile(pronoun_speaker_id, [1, c]))  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

        if self.config['order_as_features']:
            order_variable = tf.get_variable("order_feature_emb", [self.config["order_length"], self.config["feature_size"]])
            order_emb = tf.gather(order_variable, order_features)  # [k, ?, emb]
            order_emb = tf.tile(order_emb, [1, c, 1])  # [k, c, emb]
            feature_emb_list.append(order_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(
                tf.nn.relu(tf.tile(pronoun_speaker_id, [1, c]) - candidate_NP_offsets))  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                antecedent_distance_buckets)  # [c, emb]
            feature_emb_list.append(antecedent_distance_emb)

        if self.config['knowledge_as_feature']:
            number_emb = tf.gather(tf.get_variable("number_emb", [2, self.config["feature_size"]]),
                                   number_features)  # [k, c, feature_size]
            feature_emb_list.append(number_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]  [?, ?, 40]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.tile(pronoun_emb, [1, c, 1])  # [k, c, emb]
        similarity_emb = candidate_NPs_emb * target_emb  # [k, c, emb]

        pair_emb = tf.concat([target_emb, candidate_NPs_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [c]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(text_emb)[0]

        current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

        for layer in range(self.config["contextualization_layers"]):
            with tf.variable_scope("layer_{}".format(layer)):
                with tf.variable_scope("fw_cell"):
                    cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                with tf.variable_scope("bw_cell"):
                    cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                              cell_bw=cell_bw,
                                                                              inputs=current_inputs,
                                                                              sequence_length=text_len,
                                                                              initial_state_fw=state_fw,
                                                                              initial_state_bw=state_bw)

                text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
                if layer > 0:
                    # [num_sentences, max_sentence_length, emb]
                    highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2)))
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                current_inputs = text_outputs

        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def load_eval_data(self):
        print('load eval data from:', self.config["eval_path"])
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_pronoun_example(example, is_training=False), example

            with open(self.config["eval_path"]) as f:
                self.eval_data = [load_line(l) for l in f.readlines()]
            num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, session, evaluation_data=None, official_stdout=False, mode='train', title_map=None):
        if evaluation_data:
            separate_data = list()
            for tmp_example in evaluation_data:
                tensorized_example = self.tensorize_pronoun_example(tmp_example, is_training=False)
                separate_data.append((tensorized_example, tmp_example))
        else:
            separate_data = self.eval_data

        all_coreference = 0
        predict_coreference = 0
        corrct_predict_coreference = 0

        prediction_result = list()
        for example_num, (tensorized_example, example) in enumerate(separate_data):
            prediction_result_by_example = list()
            all_sentence = list()

            doc_id = example['doc_key']
            if mode == 'test' or mode == 'predict':
                print(title_map[doc_id])

            for s in example['sentences']:
                all_sentence += s

            _, _, _, _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, number_features,  candidate_NP_positions, \
            pronoun_positions, name_positions, status_positions, order_features, labels, _ = tensorized_example

            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            pronoun_coref_scores = session.run(self.predictions, feed_dict=feed_dict)

            pronoun_coref_scores = pronoun_coref_scores[0]  # [4, 4]

            if self.config["use_multi_span"]:
                gold_starts = tf.squeeze(gold_starts[:, :1], 1).eval()
                gold_ends = tf.squeeze(gold_ends[:, :1], 1).eval()

            for i, pronoun_coref_scores_by_example in enumerate(pronoun_coref_scores):
                current_pronoun_index = int(pronoun_positions[i][0])
                pronoun_position_start = int(gold_starts[current_pronoun_index])
                pronoun_position_end = int(gold_ends[current_pronoun_index]) + 1

                current_pronoun = ''.join(all_sentence[pronoun_position_start:pronoun_position_end])

                pronoun_coref_scores_by_example = pronoun_coref_scores_by_example[1:]  # [1,3]

                # labels [4, 3] bool
                prediction_result_by_example.append((pronoun_coref_scores_by_example.tolist(), labels[i]))

                for j, tmp_score in enumerate(pronoun_coref_scores_by_example.tolist()):
                    current_candidate_index = int(candidate_NP_positions[i][j])
                    candidate_positions_start = int(gold_starts[current_candidate_index])
                    candidate_positions_end = int(gold_ends[current_candidate_index]) + 1
                    current_candidate = ''.join(all_sentence[candidate_positions_start:candidate_positions_end])
                    if tmp_score > 0:
                        msg = '{} link to: {} ({},{}) \t'.format(current_pronoun, current_candidate,
                                                                 candidate_positions_start, candidate_positions_end)
                        predict_coreference += 1
                        if labels[i][j]:
                            corrct_predict_coreference += 1
                            msg += 'True-predict' + '\t' + 'score: ' + str(tmp_score)
                        else:
                            msg += 'False-predict' + '\t' + 'score: ' + str(tmp_score)
                        print(msg)

                print('Label: ')
                for n, label in enumerate(labels[i]):
                    current_candidate_index = int(candidate_NP_positions[i][n])
                    candidate_positions_start = int(gold_starts[current_candidate_index])
                    candidate_positions_end = int(gold_ends[current_candidate_index]) + 1
                    current_candidate = ''.join(all_sentence[candidate_positions_start:candidate_positions_end])
                    if labels[i][n]:
                        label_msg = '{} link to: {} ({},{}) \t'.format(current_pronoun, current_candidate,
                                                                       candidate_positions_start,
                                                                       candidate_positions_end)
                        print(label_msg)

                for l in labels[i]:
                    if l:
                        all_coreference += 1
            prediction_result.append(prediction_result_by_example)

        summary_dict = {}
        if mode == 'predict':
            summary_dict["Average F1 (py)"] = 0
            summary_dict["Average precision (py)"] = 0
            summary_dict["Average recall (py)"] = 0
            print('there is no positive prediction')
            f1 = 0
        else:
            if predict_coreference > 0:
                p = corrct_predict_coreference / predict_coreference
                r = corrct_predict_coreference / all_coreference
                f1 = 2 * p * r / (p + r)
                summary_dict["Average F1 (py)"] = f1
                print("Average F1 (py): {:.2f}%".format(f1 * 100))
                summary_dict["Average precision (py)"] = p
                print("Average precision (py): {:.2f}%".format(p * 100))
                summary_dict["Average recall (py)"] = r
                print("Average recall (py): {:.2f}%".format(r * 100))
            else:
                summary_dict["Average F1 (py)"] = 0
                summary_dict["Average precision (py)"] = 0
                summary_dict["Average recall (py)"] = 0
                print('there is no positive prediction')
                f1 = 0

        return util.make_summary(summary_dict), f1
