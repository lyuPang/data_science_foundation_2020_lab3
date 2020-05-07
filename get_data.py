from sklearn.model_selection import train_test_split
import random
import numpy as np
import warnings
import os


# dataset, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y)


def get_data():
    '''
    :return: 返回的是图片文件名
    '''
    X=[]
    Y=[]
    data_path='Images'
    name_index={}
    i=1
    for folder in os.listdir(data_path):
        name_index[i]=folder
        file_names=os.listdir(os.path.join(data_path,folder))
        X+=file_names
        Y+=[i]*len(file_names)
        i += 1
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
    return x_train, x_test, y_train, y_test, name_index


def cross_validation(dataset, k, random_data=False):
    """
    :param dataset: x_train
    :param random_data: 是否打乱原始数据集
    :return: 验证集，每个元素是[train_set,validation_set]，返回的是它们在dataset里的索引
    """
    index = [i for i in range(len(dataset))]
    if random_data:
        random.shuffle(index)
    divided_index = []
    m = int(dataset / k)
    for i in range(0, k, m):
        if i + m >= k - m:
            divided_index.append(index[i:])
            warnings.warn(
                "The number of the data cannot be divided by {}. The number of the samples in validation set may be greater than {}/{}".format(
                    k, len(dataset), k))
        else:
            divided_index.append(index[i:i + m])
    validation = []
    for i in range(k):
        __list = [divided_index[j] for j in range(len(divided_index)) if j != i]
        train_set = [j for k in __list for j in k]
        validation_set = divided_index[i]
        validation.append([train_set, validation_set])
    return validation


if __name__=='__main__':
    get_data()