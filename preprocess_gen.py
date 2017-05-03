# -*- coding: UTF-8 -*-

'''
Created on 2017年5月2日

@author: superhy
'''
'''
pre-process from original string file to seg file like '[1,2,3]'
different from the other partly file and data handle process
'''

import time

from interface import fileProcess
from interface.embedding import word2Vec
from interface.wordSeg import singleSegEngine

import re

def extQues(textFilePath):
    ''' return the list of question string (only one is [1]) '''
    fr = open(textFilePath, 'r')
    lines = fr.readlines()
    fr.close()
    
#     pat = re.compile(r'(！|？|。)+\Z')
    pat = re.compile(r'(！！|？？|。。)+\Z')
    
    ques_ans_tuples = []
    for line in lines:
        if line.startswith('w'):
            ques_string = line.split(':')[1]
            if len(ques_string) > 2:
                ques_string = ques_string[ : ques_string.find('\n') - 1]
                ques_ans_tuples.append(pat.sub('', ques_string))  # remove the line breaks
            
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
#     ques_ans_tuples = extQues(path)
#     
#     for str in ques_ans_tuples:
#         print(str)
#     print('#----------------------------------------------------#')
#     
#     seg_queses = segQues(ques_ans_tuples)
#     for ques in seg_queses:
#         for word in ques:
#             print(word),

    zhongyi_dir_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi/'
    zhongyi_corpus_file_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi_all.txt'
    
#     prodTotalSegCorpus(zhongyi_dir_path, zhongyi_corpus_file_path)

    zhongyi_w2v_path = '/home/superhy/intent-rec-file/model_cache/gensim/zhongyi_nopos.vector'
    
    prodCorpusW2V(zhongyi_corpus_file_path, zhongyi_w2v_path)

#     string = '用艾灸条烤穴位几分钟了什么方法可以治疗啊。请问有没有什么药？谢谢？'
#     print(string)
#     p = re.compile(r'(！|？|。)+\Z')
#     string = p.sub(',', string)
#     print(string)
