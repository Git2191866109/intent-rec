# -*- coding: UTF-8 -*-

'''
Created on 2017年5月29日

@author: superhy
'''

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib import seq2seq


# def w2v_batchseqs_tensorization(corpus_tuple_part, vocab, vocab_indices, w2v_model, ques_token_len, ans_token_len):
#     # need input word2vec model for query the word embeddings
# 
#     # corpus_tuple_part is the list of tuples include pair like: (ques, ans)
#     #=========================================================================
#     # ques_token_len and ans_token_len are the max length of
#     # question sequences and answer sequences, which are setted by system
#     # the length of generated answer is expected as ans_token_len
#     #=========================================================================
# 
#     x_train = np.zeros((len(corpus_tuple_part), ques_token_len,
#                         w2v_model.vector_size), dtype=np.float)
#     y_train = np.zeros(
#         (len(corpus_tuple_part), ans_token_len, len(vocab)), dtype=np.bool)
#     for qa_index, qa_tuple in enumerate(corpus_tuple_part):
#         ques_sentence = qa_tuple[0]
#         ans_sentence = qa_tuple[1]
#         for ques_t_index, ques_token in enumerate(ques_sentence[: ques_token_len]):
#             if ques_token in vocab:
#                 x_train[qa_index, ques_t_index] = word2Vec.getWordVec(
#                     w2v_model, ques_token)
#         for ans_t_index, ans_token in enumerate(ans_sentence[: ans_token_len]):
#             if ans_token in vocab:
#                 y_train[qa_index, ans_t_index, vocab_indices[ans_token]] = 1
# 
#     return x_train, y_train


def simple_seq2seq_core(encode_seqs, decode_seqs, w2v_dim, indices_dim, reuse=False):

    with tf.variable_scope("seq2seq_model", reuse=reuse):
        with tf.variable_scope("input"):
            net_encode = tl.layers.InputLayer(
                inputs=encode_seqs, name='encode_input')
            net_decode = tl.layers.OneHotInputLayer(
                inputs=decode_seqs, name='decode_input', depth=5)
        seq2seq_net = tl.layers.Seq2Seq(net_encode_in=net_encode, net_decode_in=net_decode,
                                        cell_fn=tf.contrib.rnn.BasicLSTMCell,
                                        n_hidden=5,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        encode_sequence_length=[5],
                                        decode_sequence_length=[5],
                                        initial_state=None,
                                        dropout=None,
                                        n_layer=1,
                                        return_seq_2d=True,
                                        name='seq2seq')
        net_out = tl.layers.DenseLayer(layer=seq2seq_net,
                                       n_units=indices_dim,
                                       act=tf.identity,
                                       name='output')
#         net_out = tl.layers.TimeDistributedLayer(layer=seq2seq_net,
#                                                  layer_class=tl.layers.DenseLayer,
#                                                  args={'n_unit':indices_dim, 'act':tf.identity, 'name':'dense'},
#                                                  name='time_out')
#     y_r = tf.nn.softmax(net_out.outputs)
    y_r = net_out.outputs
    return y_r

def test_seq_out():
    train_x = [[[0.9, 0.5, 0.3, 0.8, 0.4], [0.9, 0.5, 0.3, 0.8, 0.4], [0.9, 0.5, 0.3, 0.8, 0.4]],
               [[0.9, 0.5, 0.3, 0.8, 0.4], [0.9, 0.2, 0.7, 0.8, 0.4], [0.9, 0.5, 0.3, 0.5, 0.4]],
               [[0.9, 0.8, 0.3, 0.8, 0.6], [0.1, 0.5, 0.3, 0.8, 0.4], [0.9, 0.3, 0.3, 0.8, 0.4]]]
    train_y = [[6, 4, 1],
               [6, 5, 2],
               [4, 5, 3]]
    x = tf.placeholder(tf.float32, shape=(None, 3, 5), name="input")
    y_ = tf.placeholder(tf.int32, shape=(None, 3), name="label")
    
    y = simple_seq2seq_core(x, y_, 5, 5)
    
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        res = sess.run(y, feed_dict={x: train_x, y_: train_y})
        
        print(res)

if __name__ == '__main__':
    test_seq_out()
