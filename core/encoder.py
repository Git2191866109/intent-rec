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
    
    def addExtVecs(tagIndex, j):
        random_v = (random.randint(15, 60) * 1.0 / 100) * varyDecay
        varyVec = varyRange - varyDecay * (j - 1) - random_v
        varydirtVec = numpy.asarray(list(1 if tagVecs[tagIndex][i] > attVec[i] else -1 for i in range(len(attVec))))
        extVecs.append(attVec + varyVec * varydirtVec)
        
    if len(tagVecs) == 1: 
        for j in range(extNum):
            addExtVecs(0, j)
    elif len(tagVecs) == 2:
        for j in range(extNum):
            addExtVecs(j % 2, j)
    else:
        warnings.warn('nb_tagVecs exceed the limit! use first-2')
        for j in range(extNum):
            addExtVecs(j % 2, j)
            
    return extVecs

def seqUniDirtExt(vector_seqs, attention_seqs):
    pass

def seqBiDirtExt(gensimW2VModel, sentences, vector_seqs, attention_seqs, attention_T=0.5, ext_lemda=0.2):
    len_vectorSeqs = len(vector_seqs)
    len_attentionSeqs = len(attention_seqs)
    if len_attentionSeqs != len_vectorSeqs:
        warnings.warn('given incompatible dim_sequences!')
        
    # load all attSimVecDic firstly
    attSimVecDic = loadAttSimVecDic(gensimW2VModel, sentences, attention_seqs, attention_T)
    del(gensimW2VModel) # release the memory space
    
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
                extIndexes.append(att_i)
        extIndexes = numpy.asarray(extIndexes)
        
        org_vec_seq = vector_seqs[i]
        # doing the extension
        for j in range(len(extIndexes)):
            if extIndexes[j] == 0:
                # check index on left border, only extension half vectors on right direction
                if sentences[i][extIndexes[j]] not in attSimVecDic.keys():
                    break
                attVec = vector_seqs[i][extIndexes[j]]
                simVec = attSimVecDic[sentences[i][extIndexes[j]]]
                tagVecs = (vector_seqs[i][extIndexes[j] + 1],)
                extVecs = genExtVecs(attVec, simVec, tagVecs, (extNum + 1) / 2)
                del(attVec, simVec, tagVecs) # release the memory space
                
                for ext_i in range(len(extVecs)):
                    org_vec_seq.insert(extIndexes[j] + 1, extVecs[ext_i])
                # push the rest indexs
                extIndexes += len(extVecs)
                del(extVecs) # release the memory space
            elif extIndexes[j] == len(org_vec_seq) - 1:
                # check index on right border, only extension half vectors on left direction
                if sentences[i][extIndexes[j]] not in attSimVecDic.keys():
                    break
                attVec = vector_seqs[i][extIndexes[j]]
                simVec = attSimVecDic[sentences[i][extIndexes[j]]]
                tagVecs = (vector_seqs[i][extIndexes[j] + 1],)
                extVecs = genExtVecs(attVec, simVec, tagVecs, (extNum + 1) / 2)
                del(attVec, simVec, tagVecs) # release the memory space
                
                for ext_i in range(len(extVecs)):
                    org_vec_seq.insert(extIndexes[j] + ext_i, extVecs[ext_i])  # after insert on left, att_ele always be pushed one step
                # push the rest indexs
                extIndexes += len(extVecs)
                del(extVecs) # release the memory space
            else:
                # extension vectors on both right & left directions
                if sentences[i][extIndexes[j]] not in attSimVecDic.keys():
                    break
                attVec = vector_seqs[i][extIndexes[j]]
                simVec = attSimVecDic[sentences[i][extIndexes[j]]]
                tagVecs = (vector_seqs[i][extIndexes[j] + 1],)
                extVecs = genExtVecs(attVec, simVec, tagVecs, extNum)
                del(attVec, simVec, tagVecs) # release the memory space
                
                for ext_i in range(len(extVecs)):
                    # when insert on left, att_ele has been pushed one step
                    # so we need carefully about this, push forward the insert position
                    org_vec_seq.insert(extIndexes[j] + ext_i / 2 + (ext_i + 1) % 2, extVecs[ext_i])
                # push the rest indexs
                extIndexes += len(extVecs)
                del(extVecs) # release the memory space
        
        attExt_vec_seqs.append(org_vec_seq)
        
        # release the memory space
        del(org_vec_seq, extIndexes, extNum)
                
    return attExt_vec_seqs
    
if __name__ == '__main__':
    pass
