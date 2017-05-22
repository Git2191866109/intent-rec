# -*- coding: UTF-8 -*-

'''
Created on 2017年5月2日
@author: hylovedd
'''

from jieba import posseg
import jieba

# noun(名词)
noun = [u'n', u'nr', u'ns', u'nt', u'nz',
        u'nl', u'ng']
# time(时间词)
time = [u't', u'tg']
# place(处所词)
place = [u's']
# verb(动词)
verb = [u'v', u'vd', u'vn', u'vf', u'vx'
        u'vi', u'vl', u'vg']
# adjective(形容词)
adj = [u'a', u'ad', u'an', u'ag', u'al']
# distinguish(区别词)
dist = [u'b']
# state(状态词)
state = [u'z']
# pronoun(代词)
pronoun = [u'r', u'rr', u'ry', u'rg']
# math(数词)
math = [u'm', u'mq']
# quantifiers(量词)
quantifier = [u'q']
# adverb(副词)
adv = [u'd']
# preposition(介词)
preposition = [u'p']
# conjunction(连词)
conjunction = [u'c']
# auxiliary(助词)
auxiliary = [u'u', u'uz', u'ul', u'ug', u'ud',
             u'us', u'uj', u'uy', u'uv']
# modal(语气词)
modal = [u'e', u'y', u'o']
# symbol(符号)
symbol = [u'h', u'k', u'x', u'w']

        
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

    for word in segRes:
        print(word + ' '),
    print('')
    
#     segRes3 = singlePosSegEngine('头部')
#     print(segRes3)
