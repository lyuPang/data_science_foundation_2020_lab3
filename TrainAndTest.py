from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,accuracy_score
import sklearn
from tqdm import tqdm
import ExtractLocalDescriptors
import EncodingMethods
import get_data


if __name__=='__main__':
    x_train, x_test, y_train, y_test, name_index=get_data.get_data()
    n_train=len(x_train)
    data_x=x_train+x_test
    descriptor_list=[]
    encode_list=[]
    k=100
    # 得到描述子
    for img_name in tqdm(data_x):
        descriptor_list.append(ExtractLocalDescriptors.sift(img_name))
    # 得到编码后的数据，pca可以去掉
    for descriptors in descriptor_list:
        encoded=EncodingMethods.bag_of_word(descriptors)
        pca=PCA(n_components=k)
        encoded=pca.fit_transform(encoded)
        encode_list.append(encoded)
    #开始训练
    lin_clf = sklearn.svm.LinearSVC(max_iter=1000)
    encode_train=encode_list[:n_train]
    encode_test=encode_list[n_train:]
    lin_clf.fit(encode_train, y_train)
    #输出结果
    y_pred = lin_clf.predict(encode_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    print(f1,acc)