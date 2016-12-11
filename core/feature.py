# -*- coding: UTF-8 -*-

'''
Created on 2016年12月10日

@author: mull
'''

import os
import sys

import numpy as np
from recog import fileProcess

'''part of data load function'''

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

def load_term(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
            
    # term set, produce the dictionary by sorted indexes
    term_set_list = sorted(term_set_dict.keys())
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    
    return term_set_dict

def load_class(doc_class_list):
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

'''part of feature selection function'''
        
def MI(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    class_set_size = len(class_df_list)
    
    term_score_mat = np.log(((A + 1.0) * N) / ((A + C) * (A + B + class_set_size)))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)

    f_model = {}
    for i in range(len(term_set)):
        f_model[term_set[i]] = term_score_array[i]

    return Normalize(f_model)

def IG(class_df_list, term_set, term_class_df_mat):
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
    f_model = {}
    for i in range(len(term_set)):
        f_model[term_set[i]] = term_score_array[i]
        
    return Normalize(f_model)

def CHI(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C
    class_set_size = len(class_df_list)
    '''simplified CHI'''
    term_score_mat = np.square(A * D - C * B) / ((A + B + class_set_size) * (C + D + class_set_size))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    
    f_model = {}
    for i in range(len(term_set)):
        f_model[term_set[i]] = term_score_array[i]
        
    return Normalize(f_model)

def f_values(doc_terms_list, doc_class_list, fs_method):
    class_dict = load_class(doc_class_list)
    term_dict = load_term(doc_terms_list)
    class_df_list = stats_class_df(doc_class_list, class_dict)
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)
    term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x : x[1])]
    f_model = {}
    
    if fs_method == 'MI':
        f_model = MI(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'IG':
        f_model = IG(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'CHI':
        f_model = CHI(class_df_list, term_set, term_class_df_mat)
        
    return f_model

'''part of post processing function'''

def Normalize(f_model={}):
    maxValue = max(f_model.values())
    minValue = min(f_model.values())
    for word in f_model:
        f_model[word] = (1.0 * f_model[word] - minValue) / (maxValue - minValue)
    return f_model

def auto_attention_T(f_model, select_prop = 0.2):
    '''
    select the min value of best(select_prop) feature as attention_T
    '''

if __name__ == "__main__":
    
    filepath = fileProcess.auto_config_root() + 'exp_mid_data/sentences_labeled55000.txt'
    doc_terms_list, doc_class_list = load(filepath)
    '''input the texts list, classes list, the called method in IG, CHI and MI'''
    f_model = f_values(doc_terms_list, doc_class_list, 'CHI')


    f_model = sorted(f_model.items(), key=lambda item:item[1], reverse=False)
#     for i in f_model:
#         print i[0],' ',i[1]
    sf_model = dict(f_model[int((1 - 0.2) * len(f_model)) - 1 :])
#     print(sf_model)
    for key in sf_model.keys():
        print type(key), ': ', key, ' ', sf_model[key]
    print(len(sf_model))
