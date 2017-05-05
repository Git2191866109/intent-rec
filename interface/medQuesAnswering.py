# -*- coding: UTF-8 -*-

'''
Created on 2017年5月5日

@author: superhy
'''

def loadQuesAnsVocabData(trainFilePath, gensimW2VModelPath):
    # load file data
    fr_train = open(trainFilePath, 'r')
    trainLines = fr_train.readlines()
    fr_train.close()
    del(fr_train)
    
    corpus_tuple = []
    for line in trainLines:
        ques_line = line.split('-')[0]
        ans_line = line.split('-')[1]
        ques_words = list(word.decode('utf-8') for word in ques_line[ques_line.find('[') + 1 : ques_line.find(']')].split(','))
        ans_words = list(word.decode('utf-8') for word in ans_line[ques_line.find('[') + 1 : ans_line.find(']')].split(','))
        
        if len(ques_words) > 0 and len(ans_words) > 0:
            corpus_tuple.append((ques_words, ans_words))

if __name__ == '__main__':
    pass