# -*- coding: UTF-8 -*-

'''
Created on 2016年12月10日

@author: mull
'''

import os
import sys

import numpy as np
from recog import fileProcess

def get_term_dict(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())  # term set 排序后，按照索引做出字典
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict

def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return  class_dict

def stats_term_df(doc_terms_list, term_dict):
    term_df_dict = {}.fromkeys(term_dict.keys(), 0)
    for term in term_dict:
        for doc_terms in doc_terms_list:
            if term in doc_terms_list:
                term_df_dict[term] += 1                
    return term_df_dict

def stats_class_df(doc_class_list, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list

def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
    return  term_class_df_mat
        
def feature_selection_mi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    class_set_size = len(class_df_list)
    
    term_score_mat = np.log(((A + 1.0) * N) / ((A + C) * (A + B + class_set_size)))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)

    model = {}
    for i in range(len(term_set)):
        model[term_set[i]] = term_score_array[i]

    return Normalize(model)

def feature_selection_ig(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C
    term_df_array = np.sum(A, axis=1)
    class_set_size = len(class_df_list)
    
    p_t = term_df_array / N
    p_not_t = 1 - p_t
    p_c_t_mat = (A + 1) / (A + B + class_set_size)
    p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
    p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
    p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)
    
    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t  
    model = {}
    for i in range(len(term_set)):
        model[term_set[i]] = term_score_array[i]
        
    return Normalize(model)

def feature_selection_chi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C
    class_set_size = len(class_df_list)
    '''简化的CHI'''
    term_score_mat = np.square(A * D - C * B) / ((A + B + class_set_size) * (C + D + class_set_size))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    
    model = {}
    for i in range(len(term_set)):
        model[term_set[i]] = term_score_array[i]
        
    return Normalize(model)

def feature_selection(doc_terms_list, doc_class_list, fs_method):
    class_dict = get_class_dict(doc_class_list)
    term_dict = get_term_dict(doc_terms_list)
    class_df_list = stats_class_df(doc_class_list, class_dict)
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)
    term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x : x[1])]
    model = {}
    
    if fs_method == 'MI':
        model = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'IG':
        model = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'CHI':
        model = feature_selection_chi(class_df_list, term_set, term_class_df_mat)
        
    return model

def load(filepath=''):
    f = open(filepath)
    line = f.readline()
    doc_terms_list = []
    doc_class_list = []
    while line:
        label = line[line.find(']') + 1:-1]
        if label == '':
            label = line[line.find(']') + 1:]
        content = line[line.find('[') + 1:line.find(']')]
        content = content.split(',')
        doc_terms_list.append(content)
        doc_class_list.append(label)
        line = f.readline()
    return doc_terms_list, doc_class_list

def Normalize(model={}):
    maxValue = max(model.values())
    minValue = min(model.values())
    for word in model:
        model[word] = (1.0 * model[word] - minValue) / (maxValue - minValue)
    return model

if __name__ == "__main__":
    
    filepath = fileProcess.auto_config_root() + 'exp_mid_data/sentences_labeled55000.txt'
    doc_terms_list, doc_class_list = load(filepath)
    '''input the texts list, classes list, the called method in IG, CHI and MI'''
    model = feature_selection(doc_terms_list, doc_class_list, 'MI')


#     model=sorted(model.items(),key=lambda item:item[1],reverse=False)
#     for i in model:
#         print i[0],' ',i[1]
    for key in model:
        print key, ' ', model[key]
