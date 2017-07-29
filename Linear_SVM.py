import scipy.io
import numpy
import csv
#from sklearn import svm 
#from sklearn.metrics import confusion_matrix 
import matplotlib.pylab as plt
import math
%matplotlib inline

def Linear_SVM(N, M, train_image, train_label, C=1.00):
    X = numpy.zeros(shape=(N,784))
    #get the training samples
    random_train_images = numpy.random.choice((len(train_image[1,1,:])), size=N)
    Y = train_label[random_train_images].ravel()
    for i in range(N):
        X[i] = train_image[:,:,random_train_images[i]].ravel()
    
    #get the validation samples
    random_valid = numpy.random.choice(len(train_image[1,1,:]), size = M)
    X_valid = numpy.zeros(shape = (M, 784))
    Y_valid = train_label[random_valid].ravel()
    for i in range(M):
        X_valid[i] = train_image[:,:,random_valid[i]].ravel()
    X = numpy.asarray(X)
    X_valid = numpy.asarray(X_valid)
    
    #run the linear SVM and make the prediction
    svc = svm.SVC(C = C, kernel = 'linear').fit(X,Y)
    Y_pred = svc.predict(X_valid)
    
    #calculate prediction accuracy
    error = Y_pred - Y_valid
    num_error = numpy.count_nonzero(error)
    return (len(Y_pred) - num_error + 0.0)/(len(Y_pred))

train_sample = scipy.io.loadmat('/Users/guanhua/Desktop/hw1/data/digit-dataset/train.mat')
images = train_sample['train_images']
labels = train_sample['train_labels']
images = images.astype(float)
images = images/255

Train_set = [100, 200, 500, 1000, 2000, 5000, 10000]
count = 0
acc = numpy.zeros(shape = (7,1))
for n in Train_set:
    acc[count] = Linear_SVM(n, 10000, images, labels)
    count = count + 1

plt.plot(Train_set, acc)
plt.ylabel('Accuracy %')
plt.xlabel('Training Size')
plt.show()

