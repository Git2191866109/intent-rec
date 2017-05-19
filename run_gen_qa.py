# -*- coding: UTF-8 -*-

'''
Created on 2017年4月26日

@author: superhy
'''
import os
import time

from interface import fileProcess
from interface.medQuesAnswering import *
from interface.medQuesArgotGen import *


def test_qes2500():
    w2v_path = '/home/superhy/intent-rec-file/model_cache/gensim/med_qus-2500.vector'
    train_file_path = '/home/superhy/intent-rec-file/sentences_labeled2000.txt'
        
    corpus, words_vocab, vocab_indices, indices_vocab, w2v_model = loadSentenceVocabData(train_file_path, w2v_path)
    generator = trainTextGenerator(corpus, words_vocab, vocab_indices, w2v_model)
        
    prefix = []
    prefix.append(['颈椎/n', '生理/vn', '曲度/n', '变直/v', '症状/n', '频频/d', '理疗/n', '效果/n', '是/v', '颈椎/n'])
    prefix.append(['曾/d', '检查/vn', '纤维/n', '腺瘤/n', '左侧/f', '乳房/n', '时/ng', '有/v', '阵痛/n'])
    prefix.append(['社交/n', '恐惧症/n', '有/v', '15/m', '年/m', '人/n', '紧张/a', '没有话说/l', '焦虑/a', '想/v'])
    prefix.append(['全/a', '心/n', '扩大/v', '室间隔/n', '膜/n', '周/nr', '处/n', '回声/v', '失落/v', '断端/n'])
    prefix.append(['我/r', '儿子/n', '昨天/t', '开始/v', '手/n', '脚/n', '那里/r', '长/a', '几个/m', '水泡/n'])
    prefix.append(['现/tg', '龟头/n', '尿道口/n', '有/v', '感觉/n', '微痛/a', '我/r', '有/v', '淋病/n', '衣原体/n'])
    prefix.append(['最近/f', '心情/n', '压抑/v', '郁闷/a', '无缘无故/i', '想/v', '发火/v', '不想/v', '说话/v', '做/v'])
    prefix.append(['梅毒/n', '双阳/ns', '滴度/n', '2./m', '抗体/n', '滴度/n', '1280/m', '请问/v', '严重/a', '吗/y'])
    prefix.append(['我/r', '感觉/n', '肚子痛/n', '拉/v', '大便/d', '请/v', '告诉/v', '我/r', '原因/n', '哪个/r'])
    prefix.append(['心房颤动/i', '过速/d', '右/f', '心室肥大/n', '完全性/n', '右束/n', '支/v', '传导阻滞/n', 't/eng', '波/ns'])
        
    for i in range(10):    
        gen_context = runGenerator(generator, prefix[i], indices_vocab, w2v_model)
        
def handle_mental_w2v():
    mental_dir = '/home/superhy/intent-rec-file/mental/'
    files = os.listdir(mental_dir)
    sentences = []
    for file in files:
        sentences.extend(word2vec.LineSentence(mental_dir + file))
        
    mental_w2v_path = '/home/superhy/intent-rec-file/model_cache/gensim/mental.vector'
    start_w2v = time.clock()
    w2v_model = word2Vec.trainWord2VecModel(sentences, modelPath=mental_w2v_path)
    end_w2v = time.clock()
    
    print('train gensim word2vec model finish, use time: {0}'.format(end_w2v - start_w2v))
    print('vocab size: {0}'.format(len(w2v_model.vocab)))
    
def handle_mental_corpus():
    fileProcess.reLoadEncoding()
    
    mental_dir = '/home/superhy/intent-rec-file/mental/'
    files = os.listdir(mental_dir)
    sentences = []
    for file in files:
        sentences.extend(word2vec.LineSentence(mental_dir + file))
    
    sentences_strs = []
    for i in range(len(sentences)):
        words_str = '[' + ','.join(sentences[i]) + ']'
        sentences_strs.append(words_str)
    
    mental_all_path = '/home/superhy/intent-rec-file/mental_all.txt'
    fw = open(mental_all_path, 'w')
    fw.write('\n'.join(sentences_strs))
    fw.close()
    
def test_mental_text_generate(test_num=100):
    # load short prefix question sentences
    mental_all_path = '/home/superhy/intent-rec-file/mental_all.txt'
    fr = open(mental_all_path, 'r')
    lines = fr.readlines()
    contLength = 10
    prefix = []
    for line in lines:
        words = list(word.decode('utf-8') for word in line[line.find('[') + 1 : line.find(']')].split(','))
        if len(words) <= contLength:
            prefix.append(words)
        if len(prefix) >= test_num:
            break
    
    mental_w2v_path = '/home/superhy/intent-rec-file/model_cache/gensim/mental.vector'
    mental_generator_model_path = '/home/superhy/intent-rec-file/model_cache/mental_gen.json'
    
    corpus, words_vocab, vocab_indices, indices_vocab, w2v_model = loadSentenceVocabData(mental_all_path, mental_w2v_path)
    generator = trainTextGenerator(corpus, words_vocab, vocab_indices, w2v_model, frame_path=mental_generator_model_path)
    
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    time_str = str(time.strftime(ISOTIMEFORMAT, time.localtime())) + '\n'
    mental_res_path = '/home/superhy/intent-rec-file/mental_generator_res.txt'
    fw = open(mental_res_path, 'w')
    fw.write(time_str)
    fw.close()
    for i in range(10):    
        gen_context = runGenerator(generator, prefix[i], indices_vocab, w2v_model, res_path=mental_res_path)
        
# test_mental_text_generate()
    
def test_zhongyinopos_text_generate(test_num=2000):
    # load short prefix question sentences
    zhongyi_all_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi_all.txt'
    fr = open(zhongyi_all_path, 'r')
    lines = fr.readlines()
    contLength = 10
    prefix = []
    for line in lines:
        words = list(word.decode('utf-8') for word in line[line.find('[') + 1 : line.find(']')].split(','))
        if len(words) > 0 and len(words) <= contLength:
            prefix.append(words)
        if len(prefix) >= test_num:
            break
    
    zhongyi_w2v_path = '/home/superhy/intent-rec-file/model_cache/gensim/zhongyi_nopos.vector'
    zhongyi_generator_model_path = '/home/superhy/intent-rec-file/model_cache/zhongyi_nopos_gen.json'
    
    corpus, words_vocab, vocab_indices, indices_vocab, w2v_model = loadSentenceVocabData(zhongyi_all_path, zhongyi_w2v_path)
#     for word in w2v_model.vocab.keys():
#         print(word)
    generator = trainTextGenerator(corpus, words_vocab, vocab_indices, w2v_model, frame_path=zhongyi_generator_model_path)
     
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    time_str = str(time.strftime(ISOTIMEFORMAT, time.localtime())) + '\n'
    zhongyi_res_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi_generator_res.txt'
    fw = open(zhongyi_res_path, 'w')
    fw.write(time_str)
    fw.close()
    for i in range(test_num):    
        gen_context = runGenerator(generator, prefix[i], indices_vocab, w2v_model, res_path=zhongyi_res_path)
        
# test_zhongyinopos_text_generate()

def test_zhongyinopos_qa(test_num=5000):
    # load short question question sentences
#     zhongyi_qa_all_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi_qa_all20000.txt'
    zhongyi_qa_all_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi_qa_all50000.txt'
    fr = open(zhongyi_qa_all_path, 'r')
    lines = fr.readlines()
    question = []
    for line in lines:
        ques_line = line.split('-')[0]
        ques_words = list(word.decode('utf-8') for word in ques_line[ques_line.find('[') + 1 : ques_line.find(']')].split(','))
        if len(ques_words) > 50:
            continue
        question.append(ques_words)
        if len(question) >= test_num:
            break
    
#     zhongyi_qa_w2v_path = '/home/superhy/intent-rec-file/model_cache/gensim/zhongyi_qa_nopos20000.vector'
    zhongyi_qa_w2v_path = '/home/superhy/intent-rec-file/model_cache/gensim/zhongyi_qa_nopos50000.vector'
#     zhongyi_qa_model_path = '/home/superhy/intent-rec-file/model_cache/zhongyi_qa_nopos20000.json'
    zhongyi_qa_model_path = '/home/superhy/intent-rec-file/model_cache/zhongyi_qa_nopos50000.json'
    
    corpus_tuple, words_vocab, vocab_indices, indices_vocab, w2v_model, ques_token_len, ans_token_len = loadQuesAnsVocabData(zhongyi_qa_all_path, zhongyi_qa_w2v_path)
#     for word in w2v_model.vocab.keys():
#         print(word)
    generator = trainQuesAnsChatbot(corpus_tuple,
                                    words_vocab, vocab_indices, w2v_model,
                                    ques_token_len, ans_token_len,
                                    frame_path=zhongyi_qa_model_path)
     
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    time_str = str(time.strftime(ISOTIMEFORMAT, time.localtime())) + '\n'
#     zhongyi_qa_res_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi_qa_res20000.txt'
    zhongyi_qa_res_path = '/home/superhy/intent-rec-file/fenke_org/zhongyi_qa_res50000-5.19gpu-test.txt'
    fw = open(zhongyi_qa_res_path, 'w')
    fw.write(time_str)
    fw.close()
    for i in range(test_num):
        token_len = max(ques_token_len, ans_token_len)
        ans_context = runChatbot(generator, question[i],
                                 indices_vocab, w2v_model,
                                 token_len,
                                 res_path=zhongyi_qa_res_path)
        
test_zhongyinopos_qa()
