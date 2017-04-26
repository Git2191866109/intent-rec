# -*- coding: UTF-8 -*-

'''
Created on 2017年4月26日

@author: superhy
'''
from interface.medQuesArgotGen import *


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
