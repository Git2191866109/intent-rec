# -*- coding: UTF-8 -*-

'''
Created on 2016年12月2日

@author: superhy
'''
import warnings
import numpy

def loadAttSimVecDic(gensimW2VModel, sentences, attention_seqs, attention_T):
    attSimVecDic = {}
    if len(sentences) != len(attention_seqs):
        warnings.warn('given incompatible dim_sentences!')

def seqUniDirtExt(vector_seqs, attention_seqs):
    pass

def seqBiDirtExt(vector_seqs, attention_seqs, attention_T=0.5, ext_lemda=0.2):
    len_vectorSeqs = len(vector_seqs)
    len_attentionSeqs = len(attention_seqs)
    if len_attentionSeqs != len_vectorSeqs:
        warnings.warn('given incompatible sequences!')
    
    # count the average value of vector sequence's length
    avelen_vecSeq = numpy.mean(list(len(vecSeq) for vecSeq in vector_seqs))
    extLen_b = ext_lemda * avelen_vecSeq
    
    attExt_vec_seqs = []
    for i in range(len_vectorSeqs):
        # count the extension range from extension length base
        extLen = extLen_b * avelen_vecSeq * 1.0 / len(vector_seqs[i])
        
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