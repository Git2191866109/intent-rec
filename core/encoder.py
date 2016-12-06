# -*- coding: UTF-8 -*-

'''
Created on 2016年12月2日

@author: superhy
'''
import warnings
import numpy
from recog.embedding import word2Vec
import random

def loadAttSimVecDic(gensimW2VModel, sentences, attention_seqs, attention_T):
    if len(sentences) != len(attention_seqs):
        warnings.warn('given incompatible dim_sentences!')
        
    attSimVecDic = {}
    for i in range(len(sentences)):
        for j in range(len(attention_seqs[i])):
            if attention_seqs[i][j] > attention_T:
                if sentences[i][j] not in attSimVecDic.keys() and sentences[i][j].decode('utf-8') in gensimW2VModel.vocab:
                    simWord = word2Vec.queryMostSimWords(gensimW2VModel, sentences[i][j], topN=1)[0][0]
                    simVec = word2Vec.getWordVec(gensimW2VModel, simWord)
                    attSimVecDic[sentences[i][j]] = simVec
                    
    return attSimVecDic

def genExtVecs(attVec, simVec, tagVecs, extNum):
    '''
    @param @tagVecs: tuple: (1, ) in seqUniDirtExt or (1, 2) which right-1 left-2 in seqBiDirtExt
    '''
    N_1 = 1.0 / 3
    ''' numpy array calculation '''
    varyRange = (attVec - simVec) * N_1
    # vary from big to small
    varyDecay = varyRange * 1.0 / extNum
    extVecs = []
    if len(tagVecs) == 1: 
        for i in range(extNum):
            random_v = (random.randint(15, 60) * 1.0 / 100) * varyDecay
            varyVec = varyRange - varyDecay * (i - 1) - random_v
            varydirtVec = numpy.asarray(list(1 if tagVecs[0][i] > attVec[i] else -1 for i in range(len(attVec))))
            extVecs.append(attVec + varyVec * varydirtVec)
    elif len(tagVecs) == 2:
        for i in range(extNum):
            random_v = (random.randint(15, 60) * 1.0 / 100) * varyDecay
            varyVec = varyRange - varyDecay * (i - 1) - random_v
            varydirtVec = numpy.asarray(list(1 if tagVecs[i % 2][i] > attVec[i] else -1 for i in range(len(attVec))))
            extVecs.append(attVec + varyVec * varydirtVec)
    else:
        warnings.warn('nb_tagVecs exceed the limit! use first-2')
        for i in range(extNum):
            random_v = (random.randint(15, 60) * 1.0 / 100) * varyDecay
            varyVec = varyRange - varyDecay * (i - 1) - random_v
            varydirtVec = numpy.asarray(list(1 if tagVecs[i % 2][i] > attVec[i] else -1 for i in range(len(attVec))))
            extVecs.append(attVec + varyVec * varydirtVec)
            
    return extVecs

def seqUniDirtExt(vector_seqs, attention_seqs):
    pass

def seqBiDirtExt(vector_seqs, attention_seqs, attention_T=0.5, ext_lemda=0.2):
    len_vectorSeqs = len(vector_seqs)
    len_attentionSeqs = len(attention_seqs)
    if len_attentionSeqs != len_vectorSeqs:
        warnings.warn('given incompatible dim_sequences!')
    
    # count the average value of vector sequence's length
    avelen_vecSeq = numpy.mean(list(len(vecSeq) for vecSeq in vector_seqs))
    extNum_b = ext_lemda * avelen_vecSeq
    
    attExt_vec_seqs = []
    for i in range(len_vectorSeqs):
        # count the extension range from extension length base
        extNum = int(extNum_b * avelen_vecSeq / len(vector_seqs[i])) if extNum_b * avelen_vecSeq * 1.0 / len(vector_seqs[i]) > 1 else 1
        
        # record the elements' indexes which need extension
        extIndexes = []
        for att_i in range(len(attention_seqs[i])):
            if attention_seqs[i][att_i] > attention_T:
                extIndexes.append(att_i + 1)
        
        org_vec_seq = vector_seqs[i]
        # doing the extension
        for index in extIndexes:
            pass
    
if __name__ == '__main__':
    pass
