# -*- coding: UTF-8 -*-

'''
Created on 2017年5月3日

@author: superhy
'''

import time

from interface import fileProcess
from interface.embedding import word2Vec
from interface.wordSeg import singleSegEngine

import re

def extQues(textFilePath):
    ''' return the list of question_answer string tuples (only one is [(q, a)]) '''
    fr = open(textFilePath, 'r')
    lines = fr.readlines()
    fr.close()
    
#     pat = re.compile(r'(！|？|。)+\Z')
    pat = re.compile(r'(！！|？？|。。)+\Z')
    
    ques_ans_tuples = []
    ques_string = ''
    for line in lines:
        if line.startswith('w') and len(line.split(':')[1]) > 2:
            ques_string = line.split(':')[1]
            ques_string = pat.sub('', ques_string[ : ques_string.find('\n') - 1])
        if line.startswith('d'):
            if ques_string != '' and len(line.split(':')[1]) > 2:
                ans_string = line.split(':')[1]
                ans_string = pat.sub('', ans_string[ : ans_string.find('\n') - 1])
                ques_ans_tuples.append((ques_string, ans_string))
        
    return ques_ans_tuples

def segQues(quesStrs):
    ''' 
    seg the question strings by jieba 
    return the results as double-list: [[], [], ...] 
    '''
    seg_queses = []
    for str in quesStrs:
        seg_queses.append(singleSegEngine(str))
        
    return seg_queses

def prodTotalSegCorpus(orgTextsDirPath, corpusFilePath):
    '''
    produce final corpus file for generator training
    from original text files to one total file
    '''
    loadedFilesPath = fileProcess.listAllFilePathInDirectory(orgTextsDirPath)
    
    fw = open(corpusFilePath, 'a')
    for textFilePath in loadedFilesPath:
        ques_ans_tuples = extQues(textFilePath)
        seg_queses = segQues(ques_ans_tuples)
        sentences_strs = []
        for i in range(len(seg_queses)):
            words_str = '[' + ','.join(seg_queses[i]) + ']'
            print(words_str)
            sentences_strs.append(words_str)
        fw.write('\n'.join(sentences_strs))
        fw.write('\n')

        del(sentences_strs)  # clean the memory
    
    fw.close()
    
def prodCorpusW2V(totalCorpusFilePath, totalW2VPath):
    fr = open(totalCorpusFilePath, 'r')
    lines = fr.readlines()
    
    sentences = []
    for line in lines:
        line_sentence = list(word.decode('utf-8') for word in line[line.find('[') + 1 : line.find(']')].split(','))
        print(line_sentence)
        sentences.append(line_sentence)
        
    start_w2v = time.clock()
    w2v_model = word2Vec.trainWord2VecModel(sentences, modelPath=totalW2VPath)
    end_w2v = time.clock()
    
    print('train gensim word2vec model finish, use time: {0}'.format(end_w2v - start_w2v))
    print('vocab size: {0}'.format(len(w2v_model.vocab)))
    print('corpus count size: {0}'.format(w2v_model.corpus_count))

if __name__ == '__main__':
    
    path = '/home/superhy/intent-rec-file/fenke_org/zhongyi/肾虚怎么办(中医综合-中医科).txt'
    ques_ans_tuples = extQues(path)
     
    for tuple in ques_ans_tuples:
        print(tuple[0]),
        print(tuple[1])