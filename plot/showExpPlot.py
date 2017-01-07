# -*- coding: UTF-8 -*-

'''
Created on 2017年1月4日

@author: superhy
'''
from numpy import float128
import matplotlib.pyplot as plt

def transTrainingHistStrIntoList(histStr):
    histStr = histStr[histStr.find('[') + 1 : histStr.find(']')]
    histTupleStrs = list(tuple[tuple.find('(') + 1 : len(tuple) if tuple.find(')') == -1 else tuple.find(')')] for tuple in histStr.split('),'))
    
    hist1 = []
    print('load history...')
    for i in range(len(histTupleStrs)):
        histEleTuple = tuple(float128(ele) for ele in histTupleStrs[i].split(', '))
        hist1.append(histEleTuple)
        print('{0} iterate: {1}'.format(i + 1, histEleTuple))
        
    return hist1

def loadHistFileData(histPath):
    fr = open(histPath, 'r')
    lines = fr.readlines()
    fr.close()
    
    histStr = lines[0]
    resStr = lines[1]
    hist1 = transTrainingHistStrIntoList(histStr)
    resElesStr = resStr[1 : len(resStr) - 1].split(', ')
    res1 = [float128(resElesStr[0]), float128(resElesStr[1])]
    
    return hist1, res1

'''
plot function
'''

def plotLine(hist1, hist2, res1, res2):
    p_va1 = list(e[3] * 100 for e in hist1)
    p_va2 = list(e[3] * 100 for e in hist2)
    
    x = list(i + 1 for i in range(150))
    
    print(p_va1)
    print(p_va2)
    # print(p_a2)
    
#     plt.title('Attention vs Basic BiDirtGRU val_acc')
    plt.xlabel('iteration')
    plt.ylabel('acc(%)')
    
    plt.plot(x, p_va1, 'r', label='bidirtgru_val_acc')
    plt.plot(x, p_va2, 'b', label='attbidirtgru_val_acc')
    plt.axhline(y=res1[1] * 100, color='r', linestyle=':', linewidth=1)
    plt.axhline(y=res2[1] * 100, color='b', linestyle=':', linewidth=1)
     
    
    plt.show()
        
if __name__ == '__main__':
    path1 = '/home/superhy/文档/experiment/2017.1.7/5000vs/basic/RES_BiDirtGRU_Net_mat0_data2.txt'
    path2 = '/home/superhy/文档/experiment/2017.1.7/5000vs/att/RES_BiDirtGRU_Net_mat1_data2.txt'
    hist1, res1 = loadHistFileData(path1)
    hist2, res2 = loadHistFileData(path2)
    
    plotLine(hist1, hist2, res1, res2)
