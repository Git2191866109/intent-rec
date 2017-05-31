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


def simple_seq2seq_core(encode_seqs, decode_seqs, w2v_dim, indices_dim, batch_size, reuse=False):

    with tf.variable_scope("seq2seq_model", reuse=reuse):
        with tf.variable_scope("input", reuse=reuse):
            net_encode = tl.layers.InputLayer(
                inputs=encode_seqs, name='encode_input')
            net_decode = tl.layers.OneHotInputLayer(
                inputs=decode_seqs, name='decode_input', depth=indices_dim)
        net_seq2seq = tl.layers.Seq2Seq(net_encode_in=net_encode, net_decode_in=net_decode,
                                        cell_fn=tf.contrib.rnn.BasicLSTMCell,
                                        n_hidden=w2v_dim,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        encode_sequence_length=[3] * batch_size,
                                        decode_sequence_length=tl.layers.retrieve_seq_length_op3(decode_seqs, pad_val=0),
                                        initial_state=None,
                                        dropout=None,
                                        n_layer=1,
                                        return_seq_2d=True,
                                        name='seq2seq')
        net_out = tl.layers.DenseLayer(layer=net_seq2seq,
                                       n_units=indices_dim,
                                       act=tf.identity,
                                       name='output')
        net_reshape = tl.layers.ReshapeLayer(net_out, shape=[-1, 3, 6])

    y_loss = net_out.outputs
    y_r = tf.nn.softmax(net_reshape.outputs)
    y_max = tf.arg_max(input=y_r, dimension=2)
#     y_r = net_reshape.outputs
    return y_loss, y_r, y_max


def test_seq_out():
    train_x = [[[0.9, 0.9, 0.8, 0.7, 0.9], [0.59, 0.5, 0.3, 0.68, 0.4], [0.3, 0.5, 0.3, 0.25, 0.4]],
               [[0.1, 0.2, 0.3, 0.2, 0.4], [0.9, 0.2, 0.1, 0.032, 0.04], [0.5, 0.5, 0.3, 0.5, 0.67]],
               [[0.6, 0.7, 0.85, 0.678, 0.696], [0.1, 0.25, 0.113, 0.8, 0.24], [0.09, 0.33, 0.213, 0.08, 0.423]],
               [[0.1, 0.2, 0.3, 0.2, 0.4], [0.9, 0.2, 0.1, 0.032, 0.04], [0.5, 0.5, 0.3, 0.5, 0.67]],
               [[0.6, 0.7, 0.85, 0.678, 0.696], [0.1, 0.25, 0.113, 0.8, 0.24], [0.09, 0.33, 0.213, 0.08, 0.423]]]
    train_y = [[5, 3, 2],
               [1, 1, 3],
               [4, 1, 1],
               [1, 1, 3],
               [4, 1, 1]]
    target_y = [[5, 3, 2],
               [1, 1, 3],
               [4, 1, 1],
               [1, 1, 3],
               [4, 1, 1]]
    
    train_mask = tl.prepro.sequences_get_mask(train_y, pad_val=0)
    print(train_mask)

    x = tf.placeholder(tf.float32, shape=(None, 3, 5), name="input")
    y_ = tf.placeholder(tf.int64, shape=(None, 3), name="decode")
    y_t = tf.placeholder(tf.int64, shape=(None, 3), name="target")
    y_mask = tf.placeholder(tf.int64, shape=(None, 3), name="label_mask")

    y_loss, y_r, y_max = simple_seq2seq_core(x, y_, 5, 6, len(train_x))
#     e_loss = tl.cost.cross_entropy_seq(logits=y_loss, target_seqs=y_)
    e_loss = tl.cost.cross_entropy_seq_with_mask(logits=y_loss, target_seqs=y_t, input_mask=y_mask,
                                                 return_details=False, name="seq_output")
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(e_loss)
    
    init_op = tf.global_variables_initializer()

    NUM_EPOCH = 5
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
    #         E_loss = sess.run(e_loss, feed_dict={x: train_x, y_: train_y})
            Y_loss, Y_r, Y_max, E_loss, _ = sess.run([y_loss, y_r, y_max, e_loss, train_op],
                                                feed_dict={x: train_x, y_: train_y, y_t: target_y, y_mask: train_mask})

            print(Y_loss)
            print(Y_r)
            print(Y_max, Y_max[-1])
            print(E_loss)
        
        Y_p = sess.run(y_max, feed_dict={x: [train_x[0]] * 5, y_: [Y_max[-1]] * 5})
        print(Y_p)

if __name__ == '__main__':
    test_seq_out()
