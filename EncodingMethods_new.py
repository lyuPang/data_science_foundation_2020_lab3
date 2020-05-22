import sklearn
import numpy as np
from sklearn.cluster import KMeans

def VLAD(descriptors):
    pass

def bag_of_word(descriptors,cluster=128):
    model=KMeans(n_clusters=cluster,verbose=1,n_jobs=-1)
    model.fit(descriptors)
    return model
    #pass

def fisher_vector(descriptors):
    pass