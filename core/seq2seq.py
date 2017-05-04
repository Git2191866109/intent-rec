# -*- coding: UTF-8 -*-

'''
Created on 2017年5月3日

@author: superhy
'''

from interface.embedding import word2Vec
import numpy as np


def w2v_seqs_tensorization(corpus, vocab, vocab_indices, w2v_model, ques_token_len, ans_token_len):
    # need input word2vec model for query the word embeddings
    
    # corpus is the list of tuples include pair like: (ques, ans)
    #===========================================================================
    # ques_token_len and ans_token_len are the max length of
    # question sequences and answer sequences, which are setted by system
    # the length of generated answer is expected as ans_token_len
    #===========================================================================
    
    x_train = np.zeros((len(corpus), ques_token_len, w2v_model.vector_size), dtype=np.float32)
    y_train = np.zeros((len(corpus), ans_token_len, len(vocab)), dtype=np.bool)
    for qa_index, qa_tuple in enumerate(corpus):
        ques_sentence = qa_tuple[0]
        ans_sentence = qa_tuple[1]
        for ques_t_index, ques_token in enumerate(ques_sentence[ : ques_token_len]):
            if ques_token in vocab:
                x_train[qa_index, ques_t_index] = word2Vec.getWordVec(w2v_model, ques_token)
        for ans_t_index, ans_token in enumerate(ans_sentence[ : ans_token_len]):
            if ans_token in vocab:
                y_train[qa_index, ans_t_index, vocab_indices[ans_token]] = 1
                
    return x_train, y_train

if __name__ == '__main__':
    pass
