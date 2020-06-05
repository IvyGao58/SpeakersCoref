from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import math
import shutil
import sys
import json

import numpy as np
import tensorflow as tf
import pyhocon
import six
from tensorflow.contrib import rnn
from pycorenlp import StanfordCoreNLP


def initialize_from_env():
    if "GPU" in os.environ:
        set_gpus(int(os.environ["GPU"]))
    else:
        set_gpus()

    name = sys.argv[1]
    print("Running experiment: {}".format(name))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
    config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def flatten(l):
    return [item for sublist in l for item in sublist]


def set_gpus(*gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def bert_settings(config):
    return {
        "bert_size": config["bert_size"],
        "max_seq_length": config["max_seq_length"],
        "bert_config": config["bert_config"],
        "vocab_file": config["vocab_file"],
        "do_lower_case": config["do_lower_case"],
        "init_checkpoint": config["init_checkpoint"]
    }


def get_variable_via_scope(scope_lst):
    vars = []
    for sc in scope_lst:
        sc_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc)
        vars.extend(sc_variable)
    return vars


def get_sub_sentences(sentences):
    """
    return 3d sentence list, 最大子句数量，子句最长长度，子句数组-3d-list
    :param sentences:
    :return:
    """
    max_sub_num = 0
    max_sub_word = 0
    sub_sentences = list()
    for sentence in sentences:
        sub_num = 0
        tmp = list()
        sents = list()
        for token in sentence:
            if token == '，':
                tmp.append(token)
                # 找最大子句长度
                if len(tmp) > max_sub_word:
                    max_sub_word = len(tmp)

                # 检查子句长度
                if len(tmp) > 256:
                    raise ValueError(''.join(tmp) + " 长度超限。")

                sents.append([x for x in tmp])
                tmp.clear()
                sub_num += 1
            else:
                # 转为小写英文字母
                if token.encode('UTF-8').isalpha():
                    tmp.append(token.lower())
                else:
                    tmp.append(token)
        if len(tmp):
            sents.append([x for x in tmp])
            tmp.clear()
            sub_num += 1
        sub_sentences.append([x for x in sents])
        if sub_num > max_sub_num:
            max_sub_num = sub_num
    return max_sub_num, max_sub_word, sub_sentences


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=None):
    return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def highway(inputs, num_layers, dropout):
    for i in range(num_layers):
        with tf.variable_scope("highway_{}".format(i)):
            j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
            f = tf.sigmoid(f)
            j = tf.nn.relu(j)
            if dropout is not None:
                j = tf.nn.dropout(j, dropout)
            inputs = f * j + (1 - f) * inputs
    return inputs


def shape(x, dim):
    # print(x.get_shape())
    return x.get_shape()[dim].value or tf.shape(x)[dim]


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

    if len(inputs.get_shape()) == 3:
        batch_size = shape(inputs, 0)
        seqlen = shape(inputs, 1)
        emb_size = shape(inputs, 2)
        current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
        current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size])
    outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
    return outputs


def cnn2d(inputs, filter_sizes, in_channels=1, out_channels=128, ffnn_out_size=128, name=None):
    """
    :param inputs: 4d tensor [batch_size, in_height, in_width, in_channels]
    :param filter_sizes:  4d tensor [filter_height, filter_width, in_channels, out_channels]
    :param in_channels: 1
    :param out_channels: 卷积核个数
    :param name:
    :return:
    """
    batch_size = shape(inputs, 0)
    emb_size = shape(inputs, 2)  # width
    outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv_name = i if name is None else name + str(i)
        with tf.variable_scope("conv_{}".format(conv_name)):
            filters = tf.get_variable("filters_{}".format(i), [filter_size, emb_size, in_channels, out_channels])
        conv2 = tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding="SAME")  # [1, M ,N, out_channels]
        reshaped_conv2 = tf.reshape(conv2, [batch_size, -1, out_channels])
        h = tf.nn.relu(reshaped_conv2)
        pooled = tf.reduce_max(h, 1)  # [Batch, out_channel]
        outputs.append(pooled)
    concat_output = tf.concat(outputs, 1)  # [Batch, out_channels*len(filter_sizes)]
    reshape_output = tf.reshape(concat_output, [batch_size, -1])  # [Batch, len * output_channels]
    hidden_weights = tf.get_variable("con2d_dense_weights_{}".format(name),
                                     [len(filter_sizes) * out_channels, ffnn_out_size])
    hidden_bias = tf.get_variable("conv2d_dense_bias_{}".format(name), [ffnn_out_size])
    current_outputs = tf.nn.relu(tf.nn.xw_plus_b(reshape_output, hidden_weights, hidden_bias))  # [Batch, ffnn_out_size]
    return current_outputs


def cnn(inputs, filter_sizes, num_filters, name=None):
    num_words = shape(inputs, 0)
    num_chars = shape(inputs, 1)
    input_size = shape(inputs, 2)
    outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv_name = i if name is None else name + str(i)
        with tf.variable_scope("conv_{}".format(conv_name)):
            w = tf.get_variable("w", [filter_size, input_size, num_filters])
            b = tf.get_variable("b", [num_filters])
        conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID")  # [num_words, num_chars - filter_size, num_filters]
        h = tf.nn.relu(tf.nn.bias_add(conv, b))  # [num_words, num_chars - filter_size, num_filters]
        pooled = tf.reduce_max(h, 1)  # [num_words, num_filters]
        outputs.append(pooled)
    return tf.concat(outputs, 1)  # [num_words, num_filters * len(filter_sizes)]


def batch_gather(emb, indices):
    batch_size = shape(emb, 0)
    seqlen = shape(emb, 1)
    if len(emb.get_shape()) > 2:
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
    offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
    gathered = tf.gather(flattened_emb, indices + offset)  # [batch_size, num_indices, emb]
    if len(emb.get_shape()) == 2:
        gathered = tf.squeeze(gathered, 2)  # [batch_size, num_indices]
    return gathered


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1


class EmbeddingDictionary(object):
    def __init__(self, info, normalize=True, maybe_cache=None):
        self._size = info["size"]
        self._normalize = normalize
        self._path = info["path"]
        if maybe_cache is not None and maybe_cache._path == self._path:
            assert self._size == maybe_cache._size
            self._embeddings = maybe_cache._embeddings
        else:
            self._embeddings = self.load_embedding_dict(self._path)

    @property
    def size(self):
        return self._size

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = np.zeros(self.size)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
                    if len(embedding) == 1:  # first row
                        continue
                    assert len(embedding) == self.size
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict

    def __getitem__(self, key):
        embedding = self._embeddings[key]
        if self._normalize:
            embedding = self.normalize(embedding)
        return embedding

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return v


class CustomLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, batch_size, dropout):
        self._num_units = num_units
        self._dropout = dropout
        self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
        self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
        initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
        initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
        self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def initial_state(self):
        return self._initial_state

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
            c, h = state
            h *= self._dropout_mask
            concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
            i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
            i = tf.sigmoid(i)
            new_c = (1 - i) * c + i * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

    def _orthonormal_initializer(self, scale=1.0):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
            M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return params

        return _initializer

    def _block_orthonormal_initializer(self, output_sizes):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            assert len(shape) == 2
            assert sum(output_sizes) == shape[1]
            initializer = self._orthonormal_initializer()
            params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
            return params

        return _initializer


def get_pronoun_type(input_pronoun):
    for tmp_type in interested_pronouns:
        if input_pronoun in all_pronouns_by_type[tmp_type]:
            return tmp_type


def attention_layer(from_tensor,
                    to_tensor,
                    is_training=True,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=820,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    variable_name=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.
    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.
    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].
    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.
    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.
    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.
      variable_name: variable name.
    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).
    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]

    elif len(from_shape) == 2:
        if batch_size is None or from_seq_length is None or to_seq_length is None:
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_name = "query_" + variable_name if variable_name is not None else "query"
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name=query_name,
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_name = "key_" + variable_name if variable_name is not None else "key"
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name=key_name,
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_name = "value_" + variable_name if variable_name is not None else "value"
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name=value_name,
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    # if attention_scores.shape.as_list()[2] is not None:
    #     save_attenion = tf.squeeze(attention_scores, 0)
    #     json.dump(save_attenion.eval(), 'attention_probs.txt')

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def biaffine_layer(input1, input2, input_size, out_features, bias=(True, True)):
    # in1_features = input_size
    # linear_input_size = in1_features + int(bias[0])

    in2_features = input_size  # 256
    linear_output_size = out_features * (in2_features + int(bias[1]))

    batch_size = shape(input1, 0)
    len1 = shape(input1, 1)
    dim1 = shape(input1, 2)
    len2 = shape(input2, 1)
    dim2 = shape(input2, 2)

    if bias[0]:
        ones = tf.ones([batch_size, len1, 1], dtype=tf.float32, name=None)
        input1 = tf.concat((input1, ones), axis=2)
        dim1 += 1

    if bias[1]:
        ones = tf.ones([batch_size, len2, 1], dtype=tf.float32, name=None)
        input2 = tf.concat((input2, ones), axis=2)
        dim2 += 1

    input1 = tf.squeeze(input1, 0)  # dm1 -> dm2 x out

    affine_ws = tf.get_variable("biaffine_weights", [dim1, linear_output_size])
    affine = tf.nn.relu(tf.matmul(input1, affine_ws))  # 514

    affine = tf.expand_dims(affine, 0)  #  Batch, len1, out * dm2
    affine = tf.reshape(affine, [batch_size, len1 * out_features, dim2])  # batch, len1 x out_features, dm2

    input2 = tf.transpose(input2, [0, 2, 1])  # batch_size, dim2, len2

    biaffine = tf.transpose(tf.matmul(affine, input2), [0, 2, 1])
    biaffine = tf.reshape(biaffine, [batch_size, len2, len1, out_features])   # batch, len2, len1, out
    biaffine = tf.transpose(biaffine, [0, 2, 1, 3])  # batch, len1, len2, out

    # batch, len1, len2, out
    if out_features == 1:
        biaffine = tf.reshape(biaffine, [batch_size, len1, len2 * out_features])
        # biaffine = tf.nn.relu(reshaped_biaffine)

    return biaffine


def gcn_layer_simple(enc_inp, atilde, dropout=1.0):
    atilde = tf.expand_dims(atilde, 0)
    output = tf.nn.relu(tf.matmul(atilde, enc_inp))  # [1, k, k] * [1, k, 820]
    return output

    
def gcn_layer(enc_inp, Atilde_fw, dropout=1.0):
    """
    :param enc_inp: input sequence
    :param Atilde_fw: adjacent matrix of enc_inp
    :return:
    """
    input_emb_size = 820
    _internal_proj_size = 512
    _stack_dimension = 2
    _memory_dim = 512
    _memory_dim_out = 256
    _embedding_size = 512
    _hidden_layer1_size = 128
    _hidden_layer2_size = 128
    _output_size = 128

    # Dense layer before GCN
    dense_ws_gcn = tf.Variable(tf.random_uniform([input_emb_size, _internal_proj_size], 0, 0.1))
    dense_bs_gcn = tf.Variable(tf.random_uniform([_internal_proj_size], -0.1, 0.1))
    first_projection = lambda x: tf.nn.relu(tf.matmul(x, dense_ws_gcn) + dense_bs_gcn)
    hidden = tf.map_fn(first_projection, enc_inp)  # 512

    # GCN part
    Atilde_fw = tf.expand_dims(Atilde_fw, 0)  # add batch dim
    X1_fw = GCN_layer_fw(_internal_proj_size, _hidden_layer1_size, hidden, Atilde_fw)
    X3_dropout = tf.nn.dropout(X1_fw, dropout)  # [?, 1, 200]

    # # Final feedforward layers
    # Wf = tf.Variable(tf.random_uniform([_hidden_layer2_size, _output_size], 0, 0.1), name='Wf')
    # bf = tf.Variable(tf.random_uniform([_output_size], -0.1, 0.1), name='bf')
    # final_projection = lambda x: tf.matmul(x, Wf) + bf
    # outputs = tf.map_fn(final_projection, X3_dropout)  # 128
    return X3_dropout


def GCN_layer_fw(embedding_size, hidden_layer1_size, hidden, Atilde_fw):
    W0_fw = tf.Variable(tf.random_uniform([embedding_size, hidden_layer1_size], 0, 0.1), name='W0_fw')  # [512, 200]
    b0_fw = tf.Variable(tf.random_uniform([hidden_layer1_size], -0.1, 0.1), name='b0_fw')
    left_X1_projection_fw = lambda x: tf.matmul(x, W0_fw) + b0_fw
    left_X1_fw = tf.map_fn(left_X1_projection_fw, hidden)
    X1_fw = tf.nn.relu(tf.matmul(Atilde_fw, left_X1_fw))  # [1, k, k] * [1, k, 200]
    return X1_fw


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.
    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).
    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" % input_tensor.shape)
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.
    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


third_personal_pronouns = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them',
                           'They', 'it', 'It']

neutral_pronoun = ['it', 'It']

first_and_second_personal_pronouns = ['I', 'me', 'we', 'us', 'you', 'Me', 'We', 'Us', 'You']
relative_pronouns = ['that', 'which', 'who', 'whom', 'whose', 'whichever', 'whoever', 'whomever',
                     'That', 'Which', 'Who', 'Whom', 'Whose', 'Whichever', 'Whoever', 'Whomever']
demonstrative_pronouns = ['this', 'these', 'that', 'those', 'This', 'These', 'That', 'Those']
indefinite_pronouns = ['anybody', 'anyone', 'anything', 'each', 'either', 'everybody', 'everyone', 'everything',
                       'neither', 'nobody', 'none', 'nothing', 'one', 'somebody', 'someone', 'something', 'both',
                       'few', 'many', 'several', 'all', 'any', 'most', 'some',
                       'Anybody', 'Anyone', 'Anything', 'Each', 'Either', 'Everybody', 'Everyone', 'Everything',
                       'Neither', 'Nobody', 'None', 'Nothing', 'One', 'Somebody', 'Someone', 'Something', 'Both',
                       'Few', 'Many', 'Several', 'All', 'Any', 'Most', 'Some']
reflexive_pronouns = ['myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves',
                      'Myself', 'Ourselves', 'Yourself', 'Yourselves', 'Himself', 'Herself', 'Itself', 'Themselves']
interrogative_pronouns = ['what', 'who', 'which', 'whom', 'whose', 'What', 'Who', 'Which', 'Whom', 'Whose']
all_possessive_pronoun = ['my', 'your', 'his', 'her', 'its', 'our', 'your', 'their', 'mine', 'yours', 'his', 'hers',
                          'ours',
                          'yours', 'theirs', 'My', 'Your', 'His', 'Her', 'Its', 'Our', 'Your', 'Their', 'Mine', 'Yours',
                          'His', 'Hers', 'Ours', 'Yours', 'Theirs']
possessive_pronoun = ['his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']

all_pronouns_by_type = dict()
all_pronouns_by_type['first_and_second_personal'] = first_and_second_personal_pronouns
all_pronouns_by_type['third_personal'] = third_personal_pronouns
all_pronouns_by_type['possessive'] = possessive_pronoun

all_pronouns = list()
for pronoun_type in all_pronouns_by_type:
    all_pronouns += all_pronouns_by_type[pronoun_type]

all_pronouns = set(all_pronouns)

interested_pronouns = ['third_personal', 'possessive']

interested_entity_types = ['NATIONALITY', 'ORGANIZATION', 'PERSON', 'DATE', 'CAUSE_OF_DEATH', 'CITY', 'LOCATION',
                           'NUMBER', 'TITLE', 'TIME', 'ORDINAL', 'DURATION', 'MISC', 'COUNTRY', 'SET', 'PERCENT',
                           'STATE_OR_PROVINCE', 'MONEY', 'CRIMINAL_CHARGE', 'IDEOLOGY', 'RELIGION', 'URL', 'EMAIL']
