# -*- coding: UTF-8 -*-

'''
Created on 2016年11月17日

@author: superhy
'''
import os
import platform
import types
import warnings
from recognizer import cacheIndex
from cPickle import load


root_windows = 'D:\\intent-rec-file\\'
root_macos = ''
root_linux = '/home/superhy/intent-rec-file/'

def auto_config_root():

    global root_linux
    global root_macos
    global root_windows

    if platform.system() == 'Windows':
        return root_windows;
    elif platform.system() == 'Linux':
        return root_linux;
    else:
        return None

def reLoadEncoding():
        # 重新载入字符集
        import sys
        reload(sys)
        sys.setdefaultencoding('utf-8')
        
def listAllFilePathInDirectory(dirPath):
    '''
    list all file_path in a directory from dir folder
    '''
    reLoadEncoding()

    loadedFilesPath = []
    if type(dirPath) is types.StringType or type(dirPath) is types.UnicodeType:
        # dirPath is just a string
        files = os.listdir(dirPath)
        loadedFilesPath.extend(dirPath + file for file in files)
    elif type(dirPath) is types.ListType:
        # dirPath is a list which own many dir's paths
        for dir in dirPath:
            part_files = []
            part_files.extend(os.listdir(dir))
            loadedFilesPath.extend(dir + file for file in part_files)
    else:
        loadedFilesPath = None        
        warnings.warn('input dirPath type is wrong!')
    
    return loadedFilesPath

if __name__ == '__main__':
    
    trainDir = auto_config_root() + 'med_question_5000each/'
    med_qus_categories = cacheIndex.med_question_index.values()
    dirPath = []
    dirPath.extend(trainDir + category for category in med_qus_categories)
    
    loadedFilesPath = listAllFilePathInDirectory(dirPath)
    for file in loadedFilesPath:
        print(file)
    print('num: ' + str(len(loadedFilesPath)))