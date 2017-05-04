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

def extQuesAns(textFilePath):
    ''' return the list of question_answer string tuples (only one is [(q, a)]) '''
    fr = open(textFilePath, 'r')
    lines = fr.readlines()
    fr.close()
    
#     pat = re.compile(r'(！|？|。)+\Z')
    pat = re.compile(r'(！！|？？|。。)+\Z')
    
    ques_ans_tuples = []
    ques_string = ''
    for line in lines:
        if line.startswith('w') and line.find(':') != -1:
            if len(line.split(':')[1]) > 2:
                ques_string = line.split(':')[1]
                ques_string = pat.sub('', ques_string[ : ques_string.find('\n') - 1])
        if line.startswith('d') and line.find(':') != -1:
            if ques_string != '' and len(line.split(':')[1]) > 2:
                ans_string = line.split(':')[1]
                ans_string = pat.sub('', ans_string[ : ans_string.find('\n') - 1])
                ques_ans_tuples.append((ques_string, ans_string))
        
    return ques_ans_tuples

def segQuesAns(ques_ans_tuples):
    ''' 
    seg the question strings by jieba 
    return the results as double-list: [[], [], ...] 
    '''
    seg_quesans_tuples = []
    for tuples in ques_ans_tuples:
        seg_quesans_tuples.append((singleSegEngine(tuples[0]), singleSegEngine(tuples[1])))
        
    return seg_quesans_tuples

def prodTotalSegCorpus(orgTextsDirPath, corpusFilePath):
    '''
    produce final corpus file for generator training
    from original text files to one total file
    '''
    loadedFilesPath = fileProcess.listAllFilePathInDirectory(orgTextsDirPath)
    
    fw = open(corpusFilePath, 'a')
    for textFilePath in loadedFilesPath:
        ques_ans_tuples = extQuesAns(textFilePath)
        seg_quesans_tuples = segQuesAns(ques_ans_tuples)
        quesans_strs = []
        for i in range(len(seg_quesans_tuples)):
            ques_words_str = '[' + ','.join(seg_quesans_tuples[i][0]) + ']'
            ans_words_str = '[' + ','.join(seg_quesans_tuples[i][1]) + ']'
            print(ques_words_str + '-' + ans_words_str)
            quesans_strs.append(ques_words_str + '-' + ans_words_str)
        fw.write('\n'.join(quesans_strs))
        fw.write('\n')

        del(quesans_strs)  # clean the memory
    
    fw.close()
    
def prodCorpusW2V(totalCorpusFilePath, totalW2VPath):
    fr = open(totalCorpusFilePath, 'r')
    lines = fr.readlines()
    
    sentences = []
    for line in lines:
        if line.find('-') == -1:
            continue
        line_ques = line.split('-')[0]
        line_ans = line.split('-')[1]
        sentence_ques = list(word.decode('utf-8') for word in line_ques[line_ques.find('[') + 1 : line_ques.find(']')].split(','))
        sentence_ans = list(word.decode('utf-8') for word in line_ans[line_ans.find('[') + 1 : line_ans.find(']')].split(','))
        print(sentence_ques + sentence_ans)
        sentences.append(sentence_ques + sentence_ans)
        
    start_w2v = time.clock()
    w2v_model = word2Vec.trainWord2VecModel(sentences, modelPath=totalW2VPath)
    end_w2v = time.clock()
    
    print('train gensim word2vec model finish, use time: {0}'.format(end_w2v - start_w2v))
    print('vocab size: {0}'.format(len(w2v_model.vocab)))
    print('corpus count size: {0}'.format(w2v_model.corpus_count))

if __name__ == '__main__':
    
    path = '/home/superhy/intent-rec-file/fenke_org/zhongyi/肾虚怎么办(中医综合-中医科).txt'
#     ques_ans_tuples = extQuesAns(path)
#      
#     for tuple in ques_ans_tuples:
#         print(tuple[0]),
#         print(tuple[1])
    
    zhongyi_dir_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi/'
    zhongyi_corpus_file_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi_qa_all.txt'
    
#     prodTotalSegCorpus(zhongyi_dir_path, zhongyi_corpus_file_path)   

    zhongyi_w2v_path = '/home/superhy/intent-rec-file/model_cache/gensim/zhongyi_qa_nopos.vector'
    
#     prodCorpusW2V(zhongyi_corpus_file_path, zhongyi_w2v_path)
    
#     model = word2Vec.loadModelfromFile(zhongyi_w2v_path)
#     print(u'！' in model.vocab.keys())
#     for word in model.vocab.keys():
#         print(word)
    