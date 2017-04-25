# -*- coding: UTF-8 -*-

'''
Created on 2017年1月4日

@author: superhy
'''

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from numpy import float64

import matplotlib.pyplot as plt
import numpy as np
from interface import fileProcess
from matplotlib.pyplot import subplot


def transTrainingHistStrIntoList(histStr):
    histStr = histStr[histStr.find('[') + 1 : histStr.find(']')]
    histTupleStrs = list(tuple[tuple.find('(') + 1 : len(tuple) if tuple.find(')') == -1 else tuple.find(')')] for tuple in histStr.split('),'))
    
    hist1 = []
    print('load history...')
    for i in range(len(histTupleStrs)):
        histEleTuple = tuple(float64(ele) for ele in histTupleStrs[i].split(', '))
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
    resElesStr = resStr[resStr.find('[') + 1 : resStr.find(']')].split(', ')
    res1 = [float64(resElesStr[0]), float64(resElesStr[1])]
    
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
    
def plotMutilLines(histList, resList):
    p_va = []
    for hist in histList:
        p_va.append(list(e[3] * 100 for e in hist))
    
    x = list(i + 1 for i in range(150))
    
    fontsize = 14
    
    plt.xlabel('iteration', fontsize=fontsize)
    plt.ylabel('accuracy(%)', fontsize=fontsize)
    plt.xlim(-0.5, 150.5)
    
    color = ['r', 'b', 'm', 'g', 'c']
    type = ['-', '-', '--', '-', '--']
#     color = ['r', 'b', 'g', 'g']
#     type = ['-', '-', '--', '--']
    label = ['basic', 'bidecay', 'bicopy', 'unidecay', 'unicopy']
    for i in range(len(histList)):
        plt.plot(x, p_va[i], color=color[i], linestyle=type[i], linewidth=1.5, label=label[i]) # 曲线
        plt.axhline(y=resList[i][1] * 100, color=color[i], linestyle=type[i], linewidth=1) # 横线
    
    # show legend   
    plt.legend(loc='best', numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=fontsize)
    
    # show coordinate line
    plt.grid(True)
        
    plt.show()
    
def showExpDecay(n_0=1.0, ld=1.0):    
    xminorLocator   = MultipleLocator(0.1) #将x轴次刻度标签设置为0.1的倍数  
    yminorLocator   = MultipleLocator(0.04) #将此y轴次刻度标签设置为0.04的倍数
    fontsize = 14
    
    x = np.linspace(0.0, 5.0, endpoint=True)
    y_0 = n_0 * 1.0 * np.e ** (-ld * x * 1.0 / 25)
    y_1 = n_0 * 1.0 * np.e ** (-ld * x * 1.0 / 5)
    y_2 = n_0 * 1.0 * np.e ** (-ld * x * 1.0 / 1)
    y_3 = n_0 * 1.0 * np.e ** (-ld * x * 1.0 * 5)
    y_4 = n_0 * 1.0 * np.e ** (-ld * x * 1.0 * 25)
    
    plt.xlabel(r'$t$', fontsize=fontsize)
    plt.ylabel(r'$N(t)$', fontsize=fontsize)
#     plt.xlim(-0.01, 5.01)
    
    ax = subplot(111)
#     labelTex_0 = r'$N_0 e^{-\frac{1}{25} t}$'
#     labelTex_1 = r'$N_0 e^{-\frac{1}{5} t}$'
    labelTex_0 = r'$N_0 e^{-t / 25}$'
    labelTex_1 = r'$N_0 e^{-t / 5}$'
    labelTex_2 = r'$N_0 e^{- t}$'
    labelTex_3 = r'$N_0 e^{-5 t}$'
    labelTex_4 = r'$N_0 e^{-25 t}$'
    plt.plot(x,y_0,color = 'r',linestyle = '-', linewidth = 1.0, label=labelTex_0)
    plt.plot(x,y_1,color = 'g',linestyle = '-', linewidth = 1.0, label=labelTex_1)
    plt.plot(x,y_2,color = 'b',linestyle = '-', linewidth = 1.0, label=labelTex_2)
    plt.plot(x,y_3,color = 'c',linestyle = '-', linewidth = 1.0, label=labelTex_3)
    plt.plot(x,y_4,color = 'm',linestyle = '-', linewidth = 1.0, label=labelTex_4)
    
    # show legend   
    plt.legend(loc='best', numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=fontsize)
    
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    
    # show coordinate line
    plt.grid(True)
    
    plt.show()
        
if __name__ == '__main__':
#     path1 = '/home/superhy/文档/experiment/2017.1.7/2500vs/basic/RES_BiDirtGRU_Net_mat0_data2-1000.txt'
#     path2 = '/home/superhy/文档/experiment/2017.1.7/2500vs/att/RES_BiDirtGRU_Net_mat1_data2-1000.txt'
#     path3 = '/home/superhy/文档/experiment/2017.1.7/2500vs/att bicopy/RES_BiDirtGRU_Net_mat1_data2-1000.txt'
#     path4 = '/home/superhy/文档/experiment/2017.1.7/2500vs/att unidecay/RES_BiDirtGRU_Net_mat1_data2-1000.txt'
#     path5 = '/home/superhy/文档/experiment/2017.1.7/2500vs/att unicopy/RES_BiDirtGRU_Net_mat1_data2-1000.txt'

#     path1 = '/home/superhy/文档/experiment/2017.1.7/2500vs/basic/RES_BiDirtLSTM_Net_mat0_data2-1000.txt'
#     path2 = '/home/superhy/文档/experiment/2017.1.7/2500vs/att/RES_BiDirtLSTM_Net_mat1_data2-1000.txt'
#     path3 = '/home/superhy/文档/experiment/2017.1.7/2500vs/att bicopy/RES_BiDirtLSTM_Net_mat1_data2-1000.txt'
#     path4 = '/home/superhy/文档/experiment/2017.1.7/2500vs/att unidecay/RES_BiDirtLSTM_Net_mat1_data2-1000.txt'
#     path5 = '/home/superhy/文档/experiment/2017.1.7/2500vs/att unicopy/RES_BiDirtLSTM_Net_mat1_data2-1000.txt'

    network = {'cnns' : 'CNNs', 'gru' : 'GRU', 'lstm' : 'LSTM', 'bigru' : 'BiDirtGRU', 'bilstm' : 'BiDirtLSTM'}
    data = [(2500, 1000), (3500, 3500), (5000, 5000)]
    net = network['lstm']
    pick = data[0]
    
    path1 = 'D:/intent-exp/2017.1.7/{0}vs/basic/RES_{1}_Net_mat0_data2-{2}.txt'.format(pick[0], net, pick[1])
    path2 = 'D:/intent-exp/2017.1.7/{0}vs/att/RES_{1}_Net_mat1_data2-{2}.txt'.format(pick[0], net, pick[1])
    path3 = 'D:/intent-exp/2017.1.7/{0}vs/att bicopy/RES_{1}_Net_mat1_data2-{2}.txt'.format(pick[0], net, pick[1])
    path4 = 'D:/intent-exp/2017.1.7/{0}vs/att unidecay/RES_{1}_Net_mat1_data2-{2}.txt'.format(pick[0], net, pick[1])
    path5 = 'D:/intent-exp/2017.1.7/{0}vs/att unicopy/RES_{1}_Net_mat1_data2-{2}.txt'.format(pick[0], net, pick[1])
    
    hist1, res1 = loadHistFileData(path1)
    hist2, res2 = loadHistFileData(path2)
    hist3, res3 = loadHistFileData(path3)
    hist4, res4 = loadHistFileData(path4)
    hist5, res5 = loadHistFileData(path5) 
    
#     plotLine(hist1, hist2, res1, res2)
#     plotMutilLines([hist1, hist2, hist3, hist4, hist5], [res1, res2, res3, res4, res5])

    showExpDecay(n_0=1.0, ld=1)
