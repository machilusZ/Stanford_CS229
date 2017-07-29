from mnist import MNIST
import numpy as np
import math
import matplotlib.pylab as plt
import seaborn as sns

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train
    X_test = X_test
    return (X_train, labels_train), (X_test, labels_test)

def distance(A,B):
    return math.sqrt(sum((A-B)**2))

def initialCenter(dataSet,k):
    n=dataSet.shape[1]
    centers = np.zeros((k,n))
    for i in range (n):
        minvec = min(dataSet[:,i])
        maxvec = max(dataSet[:,i])
        centers[:,i] = minvec + maxvec*np.random.rand(k,)
    np.savetxt('testout.txt', centers, delimiter=',')
    return centers

def kmeans(data,k):
    m = data.shape[0]
    cluster_assign = np.zeros((m,2)) # first column is for center index, second column is for distance.
    centers = initialCenter(data,k)
    cluster_change = True
    while cluster_change:
        cluster_change = False
        for i in range (m):
            minDist = np.inf
            minIndex = -1
            for j in range (k):
                dist = distance(centers[j,:],data[i,:])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if cluster_assign[i,0] != minIndex:
                cluster_change = True
            cluster_assign[i,:] = minIndex,minDist
        for cent_index in range (k):
            PointsInCluster = data[np.nonzero(cluster_assign[:,0]==cent_index)[0]]
            centers[cent_index,:] = np.mean(PointsInCluster,axis=0)
    return centers, cluster_assign

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    center,cluster = kmeans(X_test,10)
    _,D = center.shape
    W = int(math.sqrt(D))
    assert W*W == D
    fig,axn = plt.subplots(1, 10)
    i =0
    for ax in axn.flat:
        print i
        sns.heatmap(np.reshape(center[i,:],(W,W)),ax=ax,xticklabels=False, yticklabels=False)
        i+=1
    plt.show()
