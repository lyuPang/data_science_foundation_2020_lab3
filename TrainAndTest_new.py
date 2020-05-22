from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,accuracy_score
import sklearn
from tqdm import tqdm
import ExtractLocalDescriptors
import EncodingMethods
import get_data
import numpy as np
from sklearn.svm import SVC

cluster=128#使用聚类进行编码 将每个图片转化成一个1*128的向量表示


if __name__=='__main__':
    x_train, x_test, y_train, y_test, name_index=get_data.get_data()
    print(len(x_train))
    n_train=len(x_train)
    data_x=x_train+x_test
    descriptor_list=[]
    encode_list=[]
    #k=100
    # 得到描述子
    i=0
    #np.concatenate((a,b),axis=1)
    D=[]
    image2length_train=[]
    image2length_test=[]
    #length_train={}
    for img_name in tqdm(x_train):
        i+=1
        if i==6:
            break#为了测试 这里只采用了五张图片
        #print(img_name)
        key_point,descriptors=ExtractLocalDescriptors.sift(img_name)
        #descriptor_list.append([key_point,descriptors])
        #length_train[img_name]=len(key_point)\
        image2length_train.append(len(key_point))
        D.append(descriptors)
        #print(type(descriptors.shape))
    # 得到编码后的数据，pca可以去掉
    Descriptor_train=np.concatenate(D,axis=0)#对所有图片的所有描述子组成一个巨大的n*128的矩阵 用于聚类 这里获取训练集
    # print(Descriptor.shape)
    # print(np.array(Descriptor))

    i=0
    D=[]
    length_test={}
    for img_name in tqdm(x_test):
        i+=1
        if i==6:
            break
        key_point,descriptors=ExtractLocalDescriptors.sift(img_name)
        #descriptor_list.append([key_point,descriptors])
        #length_train[img_name]=len(key_point)
        image2length_test.append(len(key_point))
        D.append(descriptors)
        #print(type(descriptors.shape))
    # 得到编码后的数据，pca可以去掉
    Descriptor_test=np.concatenate(D,axis=0)#测试集
    # print(Descriptor.shape)
    # print(np.array(Descriptor))



    codebook=EncodingMethods.bag_of_word(Descriptor_train,cluster=cluster)#使用聚类训练 生成codebook
    
    encoded_train=codebook.predict(Descriptor_train) #经过码本编码后的描述子类别
    s=0
    encoded_feature_list=[]
    for lengths in image2length_train:
        encoded_feature=[0 for j in range(cluster)]
        encoded_index=encoded_train[s:s+lengths]
        for i in range(lengths):
            encoded_feature[encoded_index[i]]+=1
        encoded_feature_list.append(encoded_feature)
        s+=lengths

    encoded_feature_train=np.array(encoded_feature_list) #经过码本编码后的128维度特征



    encoded_test=codebook.predict(Descriptor_test)#经过码本编码后的描述子类别
    s=0
    encoded_feature_list=[]
    for lengths in image2length_test:
        encoded_feature=[0 for j in range(cluster)]
        encoded_index=encoded_test[s:s+lengths]
        for i in range(lengths):
            encoded_feature[encoded_index[i]]+=1
        encoded_feature_list.append(encoded_feature)
        s+=lengths

    encoded_feature_test=np.array(encoded_feature_list) #经过码本编码后的128维度特征
    
    
    
    
    lin_clf = sklearn.svm.LinearSVC(max_iter=1000)
    lin_clf.fit(encoded_feature_train, y_train[:5])
    #输出结果
    y_pred = lin_clf.predict(encoded_feature_test)
    f1 = f1_score(y_test[:5], y_pred, average='weighted')
    acc = accuracy_score(y_test[:5], y_pred)
    print(f1,acc)




















    # for descriptors in descriptor_list:
    #     print(descriptors.shape)
    #     encoded=EncodingMethods.bag_of_word(descriptors)
    #     pca=PCA(n_components=k)
    #     encoded=pca.fit_transform(encoded)
    #     encode_list.append(encoded)
    # #开始训练
    # lin_clf = sklearn.svm.LinearSVC(max_iter=1000)
    # encode_train=encode_list[:n_train]
    # encode_test=encode_list[n_train:]
    # lin_clf.fit(encode_train, y_train)
    # #输出结果
    # y_pred = lin_clf.predict(encode_test)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # acc = accuracy_score(y_test, y_pred)
    # print(f1,acc)