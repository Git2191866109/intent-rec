# -*- coding: UTF-8 -*-

'''
Created on 2017年5月29日

@author: superhy
'''

import time

from TF_core import movie_dialogs_data
from interface.embedding import word2Vec
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.labeled_tensor import batch


# movie_dialogs_file_path = '/home/superhy/intent-rec-file/public_data/movie_dialogs.txt'
tuple_dialogs_file_path = '/home/superhy/intent-rec-file/public_data/tuple_movie_dialogs.txt'
movie_dialogs_w2v_model_path = '/home/superhy/intent-rec-file/model_cache/gensim/movie_dialogs/movie_dialogs_pb.vector'

# nb_dialogs_lines = 83097
nb_dialogs_tuples = 221616
# nb_dialogs_lines = 10000

# def load_all_dialog_lines(movie_dialogs_file_path):
#     dialogs_file = open(movie_dialogs_file_path, 'r')
#     all_dialog_lines = dialogs_file.readlines()
#     dialogs_file.close()
#     del(dialogs_file)
#  
#     return all_dialog_lines

def load_dialogs_tuples(tuple_dialogs_path):
    '''
    test file is line for tuples
    return:
        tuples for dialogs
    '''
    
    train_dialogs_file = open(tuple_dialogs_path, 'r')
    train_dialogs_lins = train_dialogs_file.readlines()
    train_dialogs_file.close()
    del(train_dialogs_file)
    
    train_dialogs_tuples = []
    for line in train_dialogs_lins:
        line = line.replace('_EOS\n', '_EOS')
        train_dialogs_tuples.append((line.split('<->')[0], line.split('<->')[1]))
    del(train_dialogs_lins)
    
    return train_dialogs_tuples

# def load_part_dialogs(all_dialog_lines, start_line_id, part_range):
#     ended = False
#     if start_line_id + part_range >= nb_dialogs_lines:
#         ended = True
#     conversations_partlines = all_dialog_lines[start_line_id: start_line_id + part_range]
#     del(all_dialog_lines)
# 
#     batch_dialogs_tuples = []
#     for conv_line in conversations_partlines:
#         dialogs_in_one_conversation = conv_line.split('<->')
#         for i in range(len(dialogs_in_one_conversation) - 1):
#             batch_dialogs_tuples.append(
#                 (dialogs_in_one_conversation[i].replace('_EOS\n', '_EOS'),
#                  dialogs_in_one_conversation[i + 1].replace('_EOS\n', '_EOS')))
#     batch_size = len(batch_dialogs_tuples)
#     del(conversations_partlines)
# 
#     return batch_dialogs_tuples, batch_size, ended

def load_part_dialogs_tuples(all_train_dialogs_tuples, start_id, batch_size):
    '''
    return:
        part of batch size test dialogs tuples
    '''
    
    ended = False
    if start_id + batch_size >= nb_dialogs_tuples:
        ended = True     
    part_train_dialogs_tuples = all_train_dialogs_tuples[start_id: start_id + batch_size]
    
    return part_train_dialogs_tuples, ended

def seq_batch_tensorization(batch_dialogs_tuples,
                            vocab2vector, vector_size,
                            vocab2index, tag_dict,
                            reverse_input=False):
    '''
    use the vl_bucket of movie dialogs movie_dialogs_data:
        [movie_dialogs_data.vl_bucket]
    the batch_dialogs_tuple looks like: [(input_seqs, output_seqs),...]
    reverse_input indicate that if the x_train input need be reversed(maybe improve the precision)

    return x_train, mask_x_train, 
        y_train, mask_y_train, y_target, mask_y_target
    '''

    _UNK = '^'  # use this symbol to act as the unknown characters

    x_train = np.zeros((len(batch_dialogs_tuples),
                        movie_dialogs_data.vl_bucket[0], vector_size), dtype=np.float32)
    mask_x_train = np.zeros((len(batch_dialogs_tuples), movie_dialogs_data.vl_bucket[0]), dtype=np.int16)
    y_train = np.zeros((len(batch_dialogs_tuples),
                        movie_dialogs_data.vl_bucket[1], vector_size), dtype=np.float32)
    mask_y_train = np.zeros((len(batch_dialogs_tuples), movie_dialogs_data.vl_bucket[1]), dtype=np.int16)
    
    y_target = np.zeros((len(batch_dialogs_tuples), movie_dialogs_data.vl_bucket[1]), dtype=np.int64)
    mask_y_target = np.zeros((len(batch_dialogs_tuples), movie_dialogs_data.vl_bucket[1]), dtype=np.int16)
    
    vocab2vector.update(tag_dict)

    for dialog_id, dialog_tuple in enumerate(batch_dialogs_tuples):
        ques_input = dialog_tuple[0]
        ans_input = dialog_tuple[1]
        # load the x_train, mask_x_train
        for ques_input_id, ques_token in enumerate(ques_input.split(' ')):
            # use the reversed input x_train
            if reverse_input == True:
                ques_input_id = len(ques_input.split(' ')) - 1 - ques_input_id
 
#             if check_in_vocab == True and ques_token not in w2v_model.vocab.keys():
#                 x_train[dialog_id, ques_input_id] = word2Vec.getWordVec(w2v_model, ques_token)
#             elif ques_token in tag_dict.keys():
#                 x_train[dialog_id, ques_input_id] = tag_dict[ques_token]
#             else:
#                 x_train[dialog_id, ques_input_id] = word2Vec.getWordVec(w2v_model, _UNK)
            try:
                x_train[dialog_id, ques_input_id] = vocab2vector[ques_token]
            except:
                assert('character not in vocab, use _UNK character: %s' % _UNK)
                x_train[dialog_id, ques_input_id] = vocab2vector[_UNK]
            
            mask_x_train[dialog_id, ques_input_id] = 1
        # load the y_train, mask_y_train
        # add the tag:[_GO] for y_train and mask_y_train first
        y_train[dialog_id, 0] = tag_dict[u'_GO']
        mask_y_train[dialog_id, 0] = 1
        for ans_input_id, ans_token in enumerate(ans_input.split(' ')[1:]):
            
#             if check_in_vocab == True and ans_token not in w2v_model.vocab.keys():
#                 y_train[dialog_id, ans_input_id + 1] = word2Vec.getWordVec(w2v_model, ans_token)
#                 y_target[dialog_id, ans_input_id] = vocab2index[ans_token]
#             elif ans_token in tag_dict.keys():
#                 y_train[dialog_id, ans_input_id + 1] = tag_dict[ans_token]
#                 y_target[dialog_id, ans_input_id] = vocab2index[ans_token]
#             else:
#                 y_train[dialog_id, ans_input_id + 1] = word2Vec.getWordVec(w2v_model, _UNK)
#                 y_target[dialog_id, ans_input_id] = vocab2index[_UNK]

            # using try... catch... to reduce 
            try:
                y_train[dialog_id, ans_input_id + 1] = vocab2vector[ans_token]
                y_target[dialog_id, ans_input_id] = vocab2index[ans_token]
            except:
                print('character not in vocab, use _UNK character: %s' % _UNK)
                y_train[dialog_id, ans_input_id + 1] = vocab2vector[_UNK]
                y_target[dialog_id, ans_input_id] = vocab2index[_UNK]
                
            mask_y_train[dialog_id, ans_input_id + 1] = 1
            mask_y_target[dialog_id, ans_input_id] = 1

    return x_train, mask_x_train, y_train, mask_y_train, y_target, mask_y_target

#------------------------------------------------------------------------------ tf function code

def simple_seq2seq_core(encode_seqs, encode_mask, decode_seqs, decode_mask,
                        w2v_dim, indices_dim, seq_len_bucket, reuse=False):
    '''
    arguments:
        indices_dim is the vocab size
        seq_len_bucket is the bucket of input and output sequences length: (len_encode, len_decode)
    '''

    with tf.variable_scope("seq2seq_model", reuse=reuse):
        with tf.variable_scope("input", reuse=reuse):
            net_encode = tl.layers.InputLayer(
                inputs=encode_seqs, name='encode_input')
            net_decode = tl.layers.InputLayer(
                inputs=decode_seqs, name='decode_input')
        net_seq2seq = tl.layers.Seq2Seq(net_encode_in=net_encode, net_decode_in=net_decode,
                                        cell_fn=tf.contrib.rnn.BasicLSTMCell,
                                        n_hidden=w2v_dim,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                        encode_sequence_length=tl.layers.retrieve_seq_length_op3(
                                            encode_mask, pad_val=0),
                                        decode_sequence_length=tl.layers.retrieve_seq_length_op3(
                                            decode_mask, pad_val=0),
                                        initial_state=None,
                                        dropout=None,
                                        n_layer=1,
                                        return_seq_2d=True,
                                        name='seq2seq')
        net_out = tl.layers.DenseLayer(layer=net_seq2seq,
                                       n_units=indices_dim,
                                       act=tf.identity,
                                       name='output')
#         net_reshape = tl.layers.ReshapeLayer(
#             net_out, shape=[-1, seq_len_bucket[1], indices_dim])

    y_loss = net_out.outputs
#     y_softmax = tf.nn.softmax(net_reshape.outputs)
#     y_max = tf.arg_max(input=y_softmax, dimension=2)
#     y_softmax = net_reshape.outputs
    # return y_loss, y_softmax, y_max
    return y_loss

def softmax_argmax_outlayer(dense_identity_outputs, indices_dim, seq_len_bucket, reuse=False):
    '''
    arguments:
        y_loss returned from seq2seq_core function 
            is a very good argument that dense_identity_outputs
    '''
    
    with tf.variable_scope("softmax_argmax_outlayer", reuse=reuse):
        with tf.variable_scope("input", reuse=reuse):
            encode_input = tl.layers.InputLayer(
                    inputs=dense_identity_outputs, name='dense_input')
        net_reshape = tl.layers.ReshapeLayer(
            encode_input, shape=[-1, seq_len_bucket[1], indices_dim])
        y_softmax = tf.nn.softmax(net_reshape.outputs)
        y_max = tf.arg_max(input=y_softmax, dimension=2)
        
    return y_max

def train_with_dialogs(session):

    NUM_EPOCH = 400
    BATCH_SIZE = 64

    LEARNING_RATE_BASE = 1.6
    MAX_GRAD_NORM = 8.0
#     train_steps = 1000
    train_steps = nb_dialogs_tuples / BATCH_SIZE + 1
    nb_iterations = train_steps * NUM_EPOCH
    LEARNING_RATE_DECAY = 0.99

#     dialogs_w2v_model = word2Vec.loadModelfromFile(
#         movie_dialogs_w2v_model_path)
    vocab2vector, vector_size = movie_dialogs_data.preload_word_vectors(movie_dialogs_w2v_model_path)
    vocab2index, index2vocab, tag_dict = movie_dialogs_data.build_w2v_vocab_index(
        movie_dialogs_w2v_model_path)
    del(index2vocab)  # do not need it

    x = tf.placeholder(tf.float32, shape=(
        None, movie_dialogs_data.vl_bucket[0], vector_size), name='x_input')
    mask_x = tf.placeholder(tf.int16, shape=(
        None, movie_dialogs_data.vl_bucket[0]), name='mask_x_input')
    y = tf.placeholder(tf.float32, shape=(
        None, movie_dialogs_data.vl_bucket[1], vector_size), name='y_input')
    mask_y = tf.placeholder(tf.int16, shape=(
        None, movie_dialogs_data.vl_bucket[1]), name='mask_y_input')
    t = tf.placeholder(tf.int64, shape=(
        None, movie_dialogs_data.vl_bucket[1]), name='y_target')
    mask_t = tf.placeholder(tf.int16, shape=(
        None, movie_dialogs_data.vl_bucket[1]), name='mask_y_target')

    # optimizer for loss
    y_loss = simple_seq2seq_core(
        x, mask_x, y, mask_y,
        vector_size,
        len(vocab2index.keys()),
        movie_dialogs_data.vl_bucket,
        reuse=False)  # (y_softmax), y_max is not used here
    loss_seq = tl.cost.cross_entropy_seq_with_mask(
        logits=y_loss, target_seqs=t, input_mask=mask_t,
        return_details=False, name="loss_seq_output")
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_step,
        decay_steps=nb_iterations,
        decay_rate=LEARNING_RATE_DECAY)
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss_seq, trainable_variables), MAX_GRAD_NORM)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)#.minimize(loss=loss_seq, global_step=global_step)
    train_op = optimizer.apply_gradients(
        zip(grads, trainable_variables), global_step=global_step)

    # load data
#     all_dialog_lines = load_all_dialog_lines(movie_dialogs_file_path)
    train_dialogs_tuples = load_dialogs_tuples(tuple_dialogs_file_path)
    
    # init all variables
    init_op = tf.global_variables_initializer()
    with session.as_default():
        session.run(init_op)
        # run epoch iterations
        for i in range(NUM_EPOCH):
            print('Epoch: {0} is running...'.format(i))
            # load all dialogs step by step until ended        
            start_id = 0
            step = 0
            start_100steps = time.clock()
            while(True):
                # part batch data
                batch_dialogs_tuples, ended = load_part_dialogs_tuples(
                    train_dialogs_tuples, start_id, BATCH_SIZE)
                x_train, mask_x_train, y_train, mask_y_train, y_target, mask_y_target = seq_batch_tensorization(
                    batch_dialogs_tuples, vocab2vector, vector_size, vocab2index, tag_dict)
                start_id += BATCH_SIZE
                    
                COST_SEQ, _ = session.run([loss_seq, train_op], feed_dict={
                    x: x_train, mask_x: mask_x_train,
                    y: y_train, mask_y: mask_y_train,
                    t: y_target, mask_t: mask_y_target})
                    
                step += 1
                if step % 100 == 0:
                    end_100steps = time.clock()
                    print('Epoch: %d -> After %d steps iterated, loss is %f, using time: %f s' % 
                          (i, step, COST_SEQ, end_100steps - start_100steps))
                    start_100steps = end_100steps
                if ended == True: break
                
    return y_loss, x, mask_x, y, mask_y
    
#------------------------------------------------------------------------------ function for test experiments (predict)

tuple_test_dialogs_path = '/home/superhy/intent-rec-file/public_data/tuple_test_dialogs.txt'
output_write_path = '/home/superhy/intent-rec-file/public_data/output/test_dialogs_output-2017.6.11.txt'

nb_test_dialogs_tuples = 600

def load_test_dialogs_tuples(tuple_test_dialogs_path):
    '''
    test file is line for tuples
    return:
        tuples for dialogs
    '''
    
    test_dialogs_file = open(tuple_test_dialogs_path, 'r')
    test_dialogs_lins = test_dialogs_file.readlines()
    test_dialogs_file.close()
    del(test_dialogs_file)
    
    test_dialogs_tuples = []
    for line in test_dialogs_lins:
        line = line.replace('_EOS\n', '_EOS')
        test_dialogs_tuples.append((line.split('<->')[0], line.split('<->')[1]))
    del(test_dialogs_lins)
    
    return test_dialogs_tuples

def load_part_test_dialogs_tuples(all_test_dialogs_tuples, start_id, batch_size):
    '''
    return:
        part of batch size test dialogs tuples
    '''
    
    ended = False
    if start_id + batch_size >= nb_test_dialogs_tuples:
        ended = True
        
    part_test_dialogs_tuples = all_test_dialogs_tuples[start_id: start_id + batch_size]
    batch_size = len(part_test_dialogs_tuples)
    
    return part_test_dialogs_tuples, batch_size, ended

def seq_test_batch_tensorization(test_batch_dialogs_tuples,
                                 vocab2vector, vector_size, tag_dict,
                                 reverse_input=False):
    '''
    use the vl_bucket of movie dialogs movie_dialogs_data:
        [movie_dialogs_data.vl_bucket]
    the batch_dialogs_tuple looks like: [(input_seqs, output_seqs),...]
    reverse_input indicate that if the x_test input need be reversed
    (at same time, the test data reverse_flag is depending the input in training progress)

    return x_test, mask_x_test, y_test, mask_y_test
    '''
    
    _UNK = '^'  # use this symbol to act as the unknown characters

    x_test = np.zeros((len(test_batch_dialogs_tuples),
                        movie_dialogs_data.vl_bucket[0], vector_size), dtype=np.float32)
    mask_x_test = np.zeros((len(test_batch_dialogs_tuples), movie_dialogs_data.vl_bucket[0]), dtype=np.int16)
    y_test = np.zeros((len(test_batch_dialogs_tuples),
                        movie_dialogs_data.vl_bucket[1], vector_size), dtype=np.float32)
#     mask_y_test = np.zeros((len(test_batch_dialogs_tuples), movie_dialogs_data.vl_bucket[1]), dtype=np.int16)
    mask_y_test = np.ones((len(test_batch_dialogs_tuples), movie_dialogs_data.vl_bucket[1]), dtype=np.int16)
    
    vocab2vector.update(tag_dict)
    
    for dialog_id, dialog_tuple in enumerate(test_batch_dialogs_tuples):
        ques_input = dialog_tuple[0]
        ans_input = dialog_tuple[1]
        # load the x_test, mask_x_test
        for ques_input_id, ques_token in enumerate(ques_input.split(' ')):
            # use the reversed input x_test
            if reverse_input == True:
                ques_input_id = len(ques_input.split(' ')) - 1 - ques_input_id
 
            # using try... catch... to reduce time, same follows
            try:
                x_test[dialog_id, ques_input_id] = vocab2vector[ques_token]
            except:
                assert('In test progress, character not in vocab, use _UNK character: %s' % _UNK)
                x_test[dialog_id, ques_input_id] = vocab2vector[_UNK]
            
            mask_x_test[dialog_id, ques_input_id] = 1
        # load the y_test, mask_y_test
        # add the tag:[_GO] for y_test and mask_y_test first
        y_test[dialog_id, 0] = tag_dict[u'_GO']
        '''
        y_test first is '_GO', the follows all 0
        mask_y_test is all 1
        '''
#         for ans_input_id, ans_token in enumerate(ans_input.split(' ')[1:]):
#             try:
#                 y_test[dialog_id, ans_input_id + 1] = vocab2vector[ans_token]
#             except:
#                 print('In test progress, character not in vocab, use _UNK character: %s' % _UNK)
#                 y_test[dialog_id, ans_input_id + 1] = vocab2vector[_UNK]
#                 
#             mask_y_test[dialog_id, ans_input_id + 1] = 1

    return x_test, mask_x_test, y_test, mask_y_test

def sample_word_from_id_seqs(seq_max_arrays, index2vocab):
    
    sampled_word_seqs = []
    for i in range(len(seq_max_arrays)):
        ansWordsSeq = []
        for id in seq_max_arrays[i]: 
            ansWordsSeq.append(index2vocab[id])
        sampled_word_seqs.append(' '.join(ansWordsSeq))
    
    return sampled_word_seqs

def test_with_dialogs(session, y_loss, x, mask_x, y, mask_y, output_write_path):
    '''
    arguments:
        y_loss is the training return, tensor of y_loss
        x, mask_x, y, mask_y is the placeholder of needed input
    attentions:
        only use after training, for 'init_op' and 'reuse'
    return: 
        the output seqs
    '''
    
    vocab2vector, vector_size = movie_dialogs_data.preload_word_vectors(movie_dialogs_w2v_model_path)
    vocab2index, index2vocab, tag_dict = movie_dialogs_data.build_w2v_vocab_index(
        movie_dialogs_w2v_model_path)
    
    test_dialogs_tuples = load_test_dialogs_tuples(tuple_test_dialogs_path)
    
#     x = tf.placeholder(tf.float32, shape=(
#         None, movie_dialogs_data.vl_bucket[0], vector_size), name='x_input')
#     mask_x = tf.placeholder(tf.int16, shape=(
#         None, movie_dialogs_data.vl_bucket[0]), name='mask_x_input')
#     y = tf.placeholder(tf.float32, shape=(
#         None, movie_dialogs_data.vl_bucket[1], vector_size), name='y_input')
#     mask_y = tf.placeholder(tf.int16, shape=(
#         None, movie_dialogs_data.vl_bucket[1]), name='mask_y_input')
    
    ''' get the y_loss and input it into softmax_argmax'''
#     y_loss = simple_seq2seq_core(
#         x, mask_x, y, mask_y,
#         vector_size,
#         len(vocab2index.keys()),
#         movie_dialogs_data.vl_bucket,
#         reuse=True)
    y_max = softmax_argmax_outlayer(y_loss, len(vocab2index.keys()),
                                    movie_dialogs_data.vl_bucket, reuse=False)
    
    TEST_BTACH_SIZE = 8
    
    output_dialogs_seqs = []
    with session.as_default():
        print('Predict testing is running...')
            # load all dialogs step by step until ended        
        start_id = 0
        while(True):
            # part batch data
            test_batch_dialogs_tuples, batch_size, ended = load_part_test_dialogs_tuples(
                test_dialogs_tuples, start_id, TEST_BTACH_SIZE)
            x_test, mask_x_test, y_test, mask_y_test = seq_test_batch_tensorization(
                test_batch_dialogs_tuples, vocab2vector, vector_size, tag_dict, reverse_input=False)
            start_id += TEST_BTACH_SIZE
                    
            Y_max = session.run(y_max, feed_dict={
                x: x_test, mask_x: mask_x_test,
                y: y_test, mask_y: mask_y_test})
            sampled_word_seqs = sample_word_from_id_seqs(Y_max, index2vocab)
            output_dialogs_seqs.extend(sampled_word_seqs)
            
            # print the batch output of prediction
            for i in range(batch_size):
                print('predict result from %d to %d, index %d:' % (
                    start_id, start_id + TEST_BTACH_SIZE, start_id + i))
                print('Input: %s\n    Example: %s\n    Output: %s' % (
                    test_batch_dialogs_tuples[i][0], test_batch_dialogs_tuples[i][1], sampled_word_seqs[i]))
            if ended == True: break
            
    output_file = open(output_write_path, 'w')
    output_file.write('\n'.join(output_dialogs_seqs))
    output_file.close()
    del(output_file)
            
    return output_dialogs_seqs

#------------------------------------------------------------------------------ run

def run():
    _sess = tf.Session()
    y_loss, x, mask_x, y, mask_y = train_with_dialogs(_sess)
    test_with_dialogs(_sess, y_loss, x, mask_x, y, mask_y, output_write_path)

#------------------------------------------------------------------------------ test code

def test_seq_out():
    train_x = [[[0.9, 0.9, 0.8, 0.7, 0.9], [0.59, 0.5, 0.3, 0.68, 0.4], [0.3, 0.5, 0.3, 0.25, 0.4]],
               [[0.1, 0.2, 0.3, 0.2, 0.4], [0.9, 0.2, 0.1, 0.032, 0.04], [0.5, 0.5, 0.3, 0.5, 0.67]],
               [[0.6, 0.7, 0.85, 0.678, 0.696], [0.1, 0.25, 0.113, 0.8, 0.24], [0.09, 0.33, 0.213, 0.08, 0.423]],
               [[0.1, 0.2, 0.3, 0.2, 0.4], [0.9, 0.2, 0.1, 0.032, 0.04], [0.5, 0.5, 0.3, 0.5, 0.67]],
               [[0.6, 0.7, 0.85, 0.678, 0.696], [0.1, 0.25, 0.113, 0.8, 0.24], [0.09, 0.33, 0.213, 0.08, 0.423]]]
    train_mask_x = [[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]
    train_x = np.asarray(train_x, dtype=np.float32)
    train_mask_x = np.asarray(train_mask_x, dtype=np.int64)
    train_y = [[[0.9, 0.9, 0.8, 0.7, 0.9], [0.59, 0.5, 0.3, 0.68, 0.4], [0.3, 0.5, 0.3, 0.25, 0.4]],
               [[0.1, 0.2, 0.3, 0.2, 0.4], [0.9, 0.2, 0.1, 0.032, 0.04], [0.5, 0.5, 0.3, 0.5, 0.67]],
               [[0.6, 0.7, 0.85, 0.678, 0.696], [0.1, 0.25, 0.113, 0.8, 0.24], [0.09, 0.33, 0.213, 0.08, 0.423]],
               [[0.1, 0.2, 0.3, 0.2, 0.4], [0.9, 0.2, 0.1, 0.032, 0.04], [0.5, 0.5, 0.3, 0.5, 0.67]],
               [[0.6, 0.7, 0.85, 0.678, 0.696], [0.1, 0.25, 0.113, 0.8, 0.24], [0.09, 0.33, 0.213, 0.08, 0.423]]]
    train_mask_y = [[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]
    train_y = np.asarray(train_y, dtype=np.float32)
    train_mask_y = np.asarray(train_mask_y, dtype=np.int64)
    target_y = [[9, 5, 2],
                [0, 1, 4],
                [6, 0, 1],
                [0, 0, 2],
                [6, 0, 0]]
    target_y = np.asarray(target_y, dtype=np.int64)

    target_mask = tl.prepro.sequences_get_mask(target_y, pad_val=0)
    print(target_mask)

    x = tf.placeholder(tf.float32, shape=(None, 3, 5), name="input")
    mask_x = tf.placeholder(tf.int64, shape=(None, 3), name='mask_input')
    y_ = tf.placeholder(tf.float32, shape=(None, 3, 5), name="decode")
    mask_y_ = tf.placeholder(tf.int64, shape=(None, 3), name="mask_decode")
    y_t = tf.placeholder(tf.int64, shape=(None, 3), name="target")
    y_mask = tf.placeholder(tf.int64, shape=(None, 3), name="label_mask")

    y_loss = simple_seq2seq_core(
        x, mask_x, y_, mask_y_, 5, 10, (3, 3), reuse=False)
#     e_loss = tl.cost.cross_entropy_seq(logits=y_loss, target_seqs=y_)
    e_loss = tl.cost.cross_entropy_seq_with_mask(logits=y_loss, target_seqs=y_t, input_mask=y_mask,
                                                 return_details=False, name="seq_output")
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=0.2).minimize(e_loss)

    init_op = tf.global_variables_initializer()

    NUM_EPOCH = 10000

    _sess = tf.Session()
    with _sess.as_default():
#         print(_sess)
        _sess.run(init_op)

        for i in range(NUM_EPOCH):
#             print("In iteration: %d" % (i + 1))
    #         E_loss = _sess.run(e_loss, feed_dict={x: train_x, y_: train_y})
            Y_loss, E_loss, _ = _sess.run(
                [y_loss, e_loss, train_op], feed_dict={
                    x: train_x, mask_x: train_mask_x, y_: train_y, mask_y_: train_mask_y, y_t: target_y, y_mask: target_mask})

#             print(Y_loss)
            if i % 100 == 0:
                print('After %d steps iterated, loss is %f' % (i, E_loss))
    print('training session finished!')
    
    test_y = [[0.9, 0.9, 0.8, 0.7, 0.9], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]
    
    y_max = softmax_argmax_outlayer(y_loss, 10, (3, 3), reuse=False)
    
    with _sess.as_default():
        Y_p = _sess.run(y_max, feed_dict={
            x: [train_x[0]] * 5, mask_x: train_mask_x, y_: [test_y] * 5, mask_y_: train_mask_y})
        print(len(Y_p))
    print('test session finished!')
    _sess.close()

def test_speed_dataloadtensor():
    
    start_id = 0
    BATCH_SIZE = 2
    
    # load data
    vocab2vector, vector_size = movie_dialogs_data.preload_word_vectors(movie_dialogs_w2v_model_path)
#     dialogs_w2v_model = word2Vec.loadModelfromFile(
#         movie_dialogs_w2v_model_path)
#     vocab = dialogs_w2v_model.vocab.keys()
    # load all words' vector in memory
    vocab2index, index2vocab, tag_dict = movie_dialogs_data.build_w2v_vocab_index(
        movie_dialogs_w2v_model_path)
    del(index2vocab)  # do not need it
    all_dialog_tuples = load_dialogs_tuples(tuple_dialogs_file_path)
    
    start_loadtensor = time.clock()
    step = 0
    while(True):
        # part batch data
        batch_dialogs_tuples, ended = load_part_dialogs_tuples(
                    all_dialog_tuples, start_id, BATCH_SIZE)
        x_train, mask_x_train, y_train, mask_y_train, y_target, mask_y_target = seq_batch_tensorization(
                    batch_dialogs_tuples, vocab2vector, vector_size, vocab2index, tag_dict)
        start_id += BATCH_SIZE
        step += 1
        if step % 100 == 0:
            print('Has loaded %d conversations' % step)
        
        if ended == True: break
    end_loadtensor = time.clock()
    
    print('Using %f s loading about %d conversations...' % (end_loadtensor - start_loadtensor, start_id))

if __name__ == '__main__':
    test_seq_out()
#     run()
