# -*- coding: UTF-8 -*-

'''
Created on 2017年6月2日

@author: superhy
'''

'''
this part is used to process the public experiment dataset
only for tf-core function

contain: cornell_movie_dialogs_corpus
'''

from gensim.models.word2vec import LineSentence
import time

from interface.semantic import wordSeg
from interface.embedding import word2Vec

import numpy as np


dialogs_corpus_data_path = '/home/superhy/intent-rec-file/public_data/'
conversations_name = 'movie_conversations.txt'
dialogs_name = 'movie_lines.txt'
clean_dialogs_name = 'movie_cleanlines.txt'

new_dialogs_name = 'movie_dialogs.txt'

movie_dialogs_w2v_model_path = '/home/superhy/intent-rec-file/model_cache/gensim/movie_dialogs/movie_dialogs_pb.vector'

vl_bucket = (16, 16)

def seg_clean_dialogs(clean_dialogs_file_path):

    clean_dialogs_file = open(clean_dialogs_file_path, 'r')
    clean_lines = clean_dialogs_file.readlines()
    clean_dialogs_file.close()
    del(clean_dialogs_file)

    seg_clean_lines = []
    start_seg = time.clock()
    for line in clean_lines:
        seg_clean_lines.append(
            ' '.join(wordSeg.simpleSegEngineforEng(line, split_sentences=False)))
    end_seg = time.clock()
    print('seg use time: {0}'.format(end_seg - start_seg))

    seg_clean_dialogs_file_path = clean_dialogs_file_path.replace(
        '.txt', '_seg.txt')
    seg_clean_dialogs_file = open(seg_clean_dialogs_file_path, 'w')
    seg_clean_dialogs_file.write('\n'.join(seg_clean_lines))
    seg_clean_dialogs_file.close()
    print('seg the clean dialogs data and write into new seg_file!')

    del(seg_clean_lines)
    del(seg_clean_dialogs_file)

def clean_dialogs_data(old_dialogs_file_path, clean_dialogs_file_path):

    old_dialogs_file = open(old_dialogs_file_path, 'r')
    clean_dialogs_file = open(clean_dialogs_file_path, 'r')
    old_lines = old_dialogs_file.readlines()
    clean_lines = clean_dialogs_file.readlines()
    old_dialogs_file.close()
    clean_dialogs_file.close()
    del(old_dialogs_file)
    del(clean_dialogs_file)

    new_lines = []
    for i in range(len(old_lines)):
        new_line = old_lines[i][: old_lines[i].rfind(
            '+++$+++') + 8] + clean_lines[i]
        new_lines.append(new_line)
    new_dialogs_file = open(old_dialogs_file_path, 'w')
    new_dialogs_file.write(''.join(new_lines))
    new_dialogs_file.close()

    print('have replace the clean dialogs lines!')

    del(new_lines)
    del(new_dialogs_file)

def reload_dialogs_check_conversations(conversations_file_path, dialogs_file_path, new_dialogs_file_path):

    conversations_file = open(conversations_file_path, 'r')
    conversations_lines = conversations_file.readlines()
    conversations_file.close()
    del(conversations_file)

    conversations_dict = {}
    for i, line in enumerate(conversations_lines):
        conv_str = line.split('+++$+++')[3]
        conv_str = conv_str[conv_str.find('[') + 1: conv_str.find(']')]
#         print(conv_str)
        conv_sentences = [sentence[1: len(sentence) - 1]
                          for sentence in conv_str.split(', ')]
        for sentence in conv_sentences:
            conversations_dict[sentence] = i

    dialogs_file = open(dialogs_file_path, 'r')
    dialogs_lines = dialogs_file.readlines()
    dialogs_file.close()
    del(dialogs_file)

    pre_dialog_id = 'L'
    dialogs_in_one_conversation = ''
    conversations_with_dialogs = []
    for line in dialogs_lines:
        dialog_id, dialog_str = [line.split(
            '+++$+++')[0], line.split('+++$+++')[4]]
        dialog_id, dialog_str = [
            dialog_id[:len(dialog_id) - 1], dialog_str[1:len(dialog_str) - 1]]
        dialog_str = dialog_str.replace('\r', '')
        dialog_str = ' '.join(dialog_str.split(' ')[: vl_bucket[0] - 5])
        dialog_str = '_GO ' + dialog_str + ' _EOS'
#         if len(dialog_str.split(' ')) > 70:
#             print(dialog_str)

#         print(pre_dialog_id, dialog_id)
        if pre_dialog_id == 'L' or conversations_dict[dialog_id] != conversations_dict[pre_dialog_id]:
            pre_dialog_id = dialog_id
            if dialogs_in_one_conversation != '':
                conversations_with_dialogs.append(dialogs_in_one_conversation)

            dialogs_in_one_conversation = dialog_str
        else:
            dialogs_in_one_conversation += '<->' + dialog_str
    # add the last one
    conversations_with_dialogs.append(dialogs_in_one_conversation)

#     for conv in conversations_with_dialogs:
#         print(conv)
#     print(len(conversations_with_dialogs))

    new_dialogs_file = open(new_dialogs_file_path, 'w')
    new_dialogs_str = '\n'.join(conversations_with_dialogs)
    new_dialogs_file.write(new_dialogs_str)
    new_dialogs_file.close()
    print('write dialogs with %d conversations' %
          len(conversations_with_dialogs))

    del(conversations_dict)
    del(conversations_with_dialogs)

def prod_dialogs_wordvec(clean_dialogs_file_path, w2v_model_path):
    # use the clean movies dialogs data is OK!
    # can use without the dialogs in one conversation

    dialogs_file = open(clean_dialogs_file_path, 'r')
    dialogs_sentences = LineSentence(dialogs_file)

    print('training wordvec of dialogs corpus...\n--with embedding_size=64, window=5, dual_core')
    start_dialogs_w2v = time.clock()
    dialogs_w2v_model = word2Vec.trainWord2VecModel(
        dialogs_sentences, w2v_model_path, Size=64, Window=5)
    end_dialogs_w2v = time.clock()
    print('word vectors of movie dialogs has build completed, with vocab_size: {0}'.format(
        len(dialogs_w2v_model.vocab)))
    print('used time: {0}'.format(end_dialogs_w2v - start_dialogs_w2v))

    return dialogs_w2v_model

def build_w2v_vocab_index(w2v_model_path, vocab2index_back_path=None):

    w2v_model = word2Vec.loadModelfromFile(w2v_model_path)
#     print([word2Vec.getWordVec(w2v_model, '?'), word2Vec.getWordVec(w2v_model, '.')])
#     print(word2Vec.getWordVec(w2v_model, 'I'))
    words_vocab = w2v_model.vocab.keys()
#     print(type(words_vocab[0]))

    vocab2index = dict((w, i + 3) for i, w in enumerate(words_vocab))
    index2vocab = dict((i + 3, w) for i, w in enumerate(words_vocab))
    vocab2index.update({u'_PAD': 0, u'_GO': 1, u'_EOS': 2})
    index2vocab.update({0: u'_PAD', 1: u'_GO', 2: u'_EOS'})

    if vocab2index_back_path != None:
        vocab_index_file = open(vocab2index_back_path, 'w')
        vocab_index_both = []
        for key in vocab2index.keys():
            vocab_index_both_str = key + ':' + \
                str(vocab2index[key]) + ',' + str(vocab2index[key]) + ':' + key
            vocab_index_both.append(vocab_index_both_str)

        vocab_index_file.write('\n'.join(vocab_index_both))
        vocab_index_file.close()
        del(vocab_index_file)

    tag_dict = {u'_PAD': np.zeros(w2v_model.vector_size, dtype=np.float32), u'_GO': np.asarray(
        [0.1] * w2v_model.vector_size, dtype=np.float32), u'_EOS': np.asarray([0.2] * w2v_model.vector_size, dtype=np.float32)}

    return vocab2index, index2vocab, tag_dict

def preload_word_vectors(movie_dialogs_w2v_model_path):
    
    start_loadvectors = time.clock()
    dialogs_w2v_model = word2Vec.loadModelfromFile(
        movie_dialogs_w2v_model_path)
    vocab2vector = {}
    vector_size = len(word2Vec.getWordVec(dialogs_w2v_model, '.'))
    for key in dialogs_w2v_model.vocab.keys():
        vocab2vector[key] = word2Vec.getWordVec(dialogs_w2v_model, key)
    end_loadvectors = time.clock()
    del(dialogs_w2v_model)
    
    print('Loaded all vectors for words in vocab, using %d s \n    words num: %d' % 
          (end_loadvectors - start_loadvectors, len(vocab2vector)))
    
    return vocab2vector, vector_size

if __name__ == '__main__':

    #     seg_clean_dialogs(dialogs_corpus_data_path + clean_dialogs_name)

    seg_clean_dialogs_file_path = dialogs_corpus_data_path + \
        clean_dialogs_name.replace('.txt', '_seg.txt')
#     clean_dialogs_data(dialogs_corpus_data_path + dialogs_name, seg_clean_dialogs_file_path)

#     reload_dialogs_check_conversations(
#         dialogs_corpus_data_path + conversations_name,
#         dialogs_corpus_data_path + dialogs_name,
#         dialogs_corpus_data_path + new_dialogs_name)

#     prod_dialogs_wordvec(seg_clean_dialogs_file_path, movie_dialogs_w2v_model_path)

    vocab2index_back_path = '/home/superhy/intent-rec-file/public_data/vocab_id_index/vocab_index.txt'
#     vocab2index, index2vocab, tag_dict = build_w2v_vocab_index(
#         movie_dialogs_w2v_model_path, vocab2index_back_path)
#     vocab2index, index2vocab, tag_dict = build_w2v_vocab_index(
#         movie_dialogs_w2v_model_path, None)
#     print(tag_dict)
