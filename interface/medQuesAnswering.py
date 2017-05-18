# -*- coding: UTF-8 -*-

'''
Created on 2017年5月5日

@author: superhy
'''
from interface.embedding import word2Vec
from K_core import basic_Seq2Seq

def loadQuesAnsVocabData(trainFilePath, gensimW2VModelPath):
    # load file data
    fr_train = open(trainFilePath, 'r')
    trainLines = fr_train.readlines()
    fr_train.close()
    del(fr_train)
    
    corpus_tuple = []
    ques_token_len = 0
    ans_token_len = 0
    for line in trainLines:
        if line.find('-') == -1:
            continue
        ques_line = line.split('-')[0]
        ans_line = line.split('-')[1]
        ques_words = list(word.decode('utf-8') for word in ques_line[ques_line.find('[') + 1 : ques_line.find(']')].split(','))
        ans_words = list(word.decode('utf-8') for word in ans_line[ans_line.find('[') + 1 : ans_line.find(']')].split(','))
        
        if len(ques_words) > 0 and len(ques_words) <= 50 and len(ans_words) > 0 and len(ans_words) <= 50:
            ques_token_len = max(ques_token_len, len(ques_words))
            ans_token_len = max(ans_token_len, len(ans_words))
            corpus_tuple.append((ques_words, ans_words))
            
    # load word vocab indices data
    gensimW2VModel = word2Vec.loadModelfromFile(gensimW2VModelPath)
    words_vocab = gensimW2VModel.vocab.keys()
    
    vocab_indices = dict((w, i) for i, w in enumerate(words_vocab))
    indices_vocab = dict((i, w) for i, w in enumerate(words_vocab))
    
    return corpus_tuple, words_vocab, vocab_indices, indices_vocab, gensimW2VModel, ques_token_len, ans_token_len

def trainQuesAnsChatbot(corpus_tuple, words_vocab, vocab_indices,
                       w2v_model,
                       ques_token_len, ans_token_len,
                       network='LSTM_core',
                       frame_path=None):
    '''
    network: the chatbot neural network K_core (only LSTM_core now)
    frame_path: the storage path of the neural network framework model
    '''
    
    generator = basic_Seq2Seq.trainer(corpus_tuple, words_vocab, vocab_indices, w2v_model, ques_token_len, ans_token_len)
    if frame_path != None:
        basic_Seq2Seq.storageGenerator(generator, frame_path)
        print('Chatbot has been stored in path: {0}.'.format(frame_path))
    
    return generator

def runChatbot(generator, ques_test_input,
                 indices_vocab,
                 w2v_model, token_len,
                 res_path=None):
    
    answeringContext = basic_Seq2Seq.chatbot(generator, ques_test_input, indices_vocab, w2v_model, token_len)
    if res_path != None:
        fw = open(res_path, 'a')
        generate_text_str = 'Question: ' + ''.join(ques_test_input) + '\n' + 'Answering: ' + ''.join(answeringContext) + '\n'
        fw.write(generate_text_str)
        fw.close()
    
    return answeringContext

if __name__ == '__main__':
    pass
