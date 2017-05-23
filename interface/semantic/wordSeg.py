# -*- coding: UTF-8 -*-

'''
Created on 2017年5月2日
@author: hylovedd
'''

from jieba import posseg
import jieba
        
def singleSegEngine(segStr, segMode='e', userDictPath=None):
    if not userDictPath == None:
        jieba.load_userdict(userDictPath)
        
    wordGenList = []
    if segMode == 'a':
        wordGenList = jieba.cut(segStr, cut_all=True)
    elif segMode == 's':
        wordGenList = jieba.cut_for_search(segStr)
    else:
        wordGenList = jieba.cut(segStr, cut_all=False)
        
    wordStr = '_'.join(wordGenList)
    wordList = wordStr.split('_')
            
    return wordList
    
def singlePosSegEngine(segStr, _userDictPath=None):
    if not _userDictPath == None:
        jieba.load_userdict(_userDictPath)
        
    wordPosGenList = posseg.cut(segStr, HMM=True)
        
    wordPosList = []
    for wordPair in wordPosGenList:
        wordPosList.append(u'/'.join(wordPair))
        
    return wordPosList
    
def linesSeger(segLines, posNeedFlag=False):
    '''
    for one multi-line corpus text(best in one file)
    '''
    segParaList = []
    if posNeedFlag == True:
        for line in segLines:
            segParaList.extend(singlePosSegEngine(line))
    else:
        for line in segLines:
            segParaList.extend(singleSegEngine(line))
        
    return segParaList
    
def serialSeger(segStrList, posNeedFlag=False):
    '''
    for multi one-line short text(best in one file)
    '''
    segParaList = []
    if posNeedFlag == True:
        for str in segStrList:
            segParaList.append(singlePosSegEngine(str))
    else:
        for str in segStrList:
            segParaList.append(singleSegEngine(str))
        
    return segParaList

if __name__ == '__main__':
        
    segRes = singleSegEngine('习近平总书记表扬小明，小明硕士毕业于中国科学院计算所，后在日本京都大学深造')
#     segRes2 = mainObj.singlePosSegEngine('习近平总书记在北京市朝阳区表扬小明，小明硕士毕业于中国科学院计算所，后在日本京都大学深造') 
    segRes2 = singlePosSegEngine('我最近参加了高校主办的北清大数据联合会中文的一系列活动')
    
#     print(' '.join(segRes2))
#     
#     for word in segRes2:
#         print(word)

    for word in segRes2:
        print(word + ' '),
    print('')
    
#     segRes3 = singlePosSegEngine('头部')
#     print(segRes3)
