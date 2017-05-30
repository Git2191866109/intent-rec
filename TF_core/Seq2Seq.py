# -*- coding: UTF-8 -*-

'''
Created on 2017年5月29日

@author: superhy
'''

from interface.embedding import word2Vec
import numpy as np
import tensorflow as tf
import tensorlayer as tl


def w2v_batchseqs_tensorization(corpus_tuple_part, vocab, vocab_indices, w2v_model, ques_token_len, ans_token_len):
    # need input word2vec model for query the word embeddings

    # corpus_tuple_part is the list of tuples include pair like: (ques, ans)
    #=========================================================================
    # ques_token_len and ans_token_len are the max length of
    # question sequences and answer sequences, which are setted by system
    # the length of generated answer is expected as ans_token_len
    #=========================================================================

    x_train = np.zeros((len(corpus_tuple_part), ques_token_len,
                        w2v_model.vector_size), dtype=np.float)
    y_train = np.zeros(
        (len(corpus_tuple_part), ans_token_len, len(vocab)), dtype=np.bool)
    for qa_index, qa_tuple in enumerate(corpus_tuple_part):
        ques_sentence = qa_tuple[0]
        ans_sentence = qa_tuple[1]
        for ques_t_index, ques_token in enumerate(ques_sentence[: ques_token_len]):
            if ques_token in vocab:
                x_train[qa_index, ques_t_index] = word2Vec.getWordVec(
                    w2v_model, ques_token)
        for ans_t_index, ans_token in enumerate(ans_sentence[: ans_token_len]):
            if ans_token in vocab:
                y_train[qa_index, ans_t_index, vocab_indices[ans_token]] = 1

    return x_train, y_train


def simple_seq2seq_core(encode_seqs, decode_seqs, w2v_dim, indices_dim, reuse=False):

    with tf.variable_scope("seq2seq_model", reuse=reuse):
        with tf.variable_scope("input"):
            net_encode = tl.layers.InputLayer(
                inputs=encode_seqs, name='encode_input')
            net_decode = tl.layers.InputLayer(
                inputs=decode_seqs, name='decode_input')
        seq2seq_net = tl.layers.Seq2Seq(net_encode_in=net_encode, net_decode_in=net_decode, 
                                        cell_fn=tf.contrib.rnn.BasicLSTMCell, 
                                        n_hidden=200, 
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        encode_sequence_length=tl.layers.retrieve_seq_length_op2(encode_seqs), 
                                        decode_sequence_length=tl.layers.retrieve_seq_length_op2(decode_seqs), 
                                        initial_state=None, 
                                        dropout=None, 
                                        n_layer=1, 
                                        return_seq_2d=False, 
                                        name='seq2seq')
        net_out = tl.layers.TimeDistributedLayer(layer=seq2seq_net,
                                                 layer_class=tl.layers.DenseLayer,
                                                 args={'n_units': indices_dim, 'act': tf.identity, 'name': 'dense'},
                                                 name='seq_output')
    y = net_out.outputs
    return y

def test_seq_out():
    pass

if __name__ == '__main__':
    pass
