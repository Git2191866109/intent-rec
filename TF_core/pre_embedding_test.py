# -*- coding: UTF-8 -*-

'''
Created on 2017年6月1日

@author: superhy
'''

import tensorflow as tf
import tensorlayer as tl


def loadEmbedding():
    vocabulary_size = 50000  # maximum number of word in vocabulary
    embedding_size = 128
    num_sampled = 64

    model_file_path = "/home/superhy/文档/code/test/tf_learn/"
    model_file_name = "wordvec/model_word2vec_50k_128"

    all_var = tl.files.load_npy_to_any(
        path=model_file_path, name=model_file_name + '.npy')

#     for key in all_var.keys():
#         print(key)

    data = all_var['data']
    count = all_var['count']
    dictionary = all_var['dictionary']
    reverse_dictionary = all_var['reverse_dictionary']
    
    print(data[:10])
    print(count[:10])

    train_inputs = tf.placeholder(tf.int64, shape=[None])
    train_labels = tf.placeholder(tf.int64, shape=[None, 1])
    emb_box = tl.layers.Word2vecEmbeddingInputlayer(
        inputs=train_inputs,
        train_labels=train_labels,
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        num_sampled=num_sampled,
        name='word2vec_layer',
    )

    print(emb_box.all_params)
    
    emb_net = tl.layers.EmbeddingInputlayer(
                inputs = train_inputs,
                vocabulary_size = vocabulary_size,
                embedding_size = embedding_size,
                name ='embedding_layer')

    _sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(_sess)
    tl.files.assign_params(_sess, [emb_box.all_params[0]], emb_net)
    
    word_id = dictionary[b'would']
    vector = _sess.run(emb_net.outputs, feed_dict={train_inputs: [word_id]})
    print(vector)

if __name__ == '__main__':
    loadEmbedding()
