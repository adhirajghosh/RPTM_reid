import torch
import os

def keyfromval(dic, val):
    return list(dic.keys())[list(dic.values()).index(val)]

def strint(x, dataset):
    if dataset =='veri':

        if len(str(x))==1:
            return '00'+str(x)
        if len(str(x))==2:
            return '0'+str(x)
        if len(str(x))==3:
            return str(x)
    
    if dataset == 'duke':
        if len(str(x))==1:
            return '000'+str(x)
        if len(str(x))==2:
            return '00'+str(x)
        if len(str(x))==3:
            return '0'+str(x)
        if len(str(x))==4:
            return str(x)
    
    if dataset == 'vehicleid':
        if len(str(x))==1:
            return '0000'+str(x)
        if len(str(x))==2:
            return '000'+str(x)
        if len(str(x))==3:
            return '00'+str(x)
        if len(str(x))==4:
            return '0'+str(x)
        if len(str(x))==5:
            return str(x)

def search1(pkl, path):
    #MAIN ONE
    start = 0
    count = 0
    end = 0
    data_index = []
    for i in range(1, 777):
        label = strint(i)
        if os.path.isdir(os.path.join(path, label)) is True:
            size = len(pkl[label])
            start = end
            end = end+size
            data_index.append((count, start, end-1))
            count+=1
        if label == '769':
            size = len(pkl[label])
            start = end
            end = end+size
            data_index.append((count, start, end-1))
            break
    return data_index

def search(path):
    #MAIN ONE
    start = 0
    count = 0
    end = 0
    data_index = []
    for i in sorted(os.listdir(path)):
        x = len(os.listdir(os.path.join(path,i)))
        data_index.append((count, start, start+x-1))
        count = count+1
        start = start+x
    return data_index

def search_index(pkl, path, folders):
    start = 0
    count = 0
    end = 0
    data_index = []
    for i in range(0, len(folders)):
        label = folders[i]
        size = len(pkl[label])
        start = end
        end = end+size
        data_index.append((count, start, end-1))
        count+=1
    return data_index


