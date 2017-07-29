import scipy.io
import scipy as sp
import numpy as np
import csv
import matplotlib.pyplot as plt

def data_std(x):
    return ((x - np.mean(x, axis=0)) / np.std(x, axis=0))

def data_log(x):
    xtrain_reg = x + 0.1
    xtrain_log = xtrain_reg
    for i in range(0, xtrain_reg.shape[1]):
        xtrain_log[i]=np.log(xtrain_reg[i])
    return xtrain_log
    
def data_bin(x): 
    xtrain_bina=np.zeros((x.shape[0],x.shape[1]))
    for m in range(0,x.shape[0]):
        for n in range(0,x.shape[1]):
            if (xtrain[m][n]<=0):
                xtrain_bina[m][n]=0
            else:
                xtrain_bina[m][n]=1
    return xtrain_bina

def LR_gd(x_train,y_train,e,lamda,iteration):
    b0 = np.zeros(57)+0.0000001
    b = b0.reshape(57,1)
    loss_gd = np.zeros(iteration)
    for i in range (0,iteration):
        b = b - e*(2*lamda*b - np.dot(x_train.T,(y_train-(1/(1+np.exp(np.dot(x_train,-b)))))))
        loss_gd[i]=lamda*np.dot(b.T,b)-np.dot(y_train.T,np.log((1/(1+np.exp(np.dot(x_train,-b))))))-np.dot(1-y_train.T,np.log(1-(1/(1+np.exp(np.dot(x_train,-b))))))
    np.savetxt('loss_gd100000.csv', loss_gd, delimiter=',') 
    plt.plot([x for x in range(1,iteration+1)],loss_gd)
    plt.xlabel("No. of Iterations")
    plt.ylabel("Trainning Loss")
    plt.title("Training Loss of Gradient Descent(Standardized data)")
    plt.show()
    return b

def LR_sgd(x_train,y_train,e,lamda,iteration):
    #b0 = np.zeros(57)+0.0000001
    b0=np.sqrt(2.0/57)*np.random.randn(57)
    b = b0.reshape(57,1)
    size = x_train.shape[0]
    loss_sgd = np.zeros(iteration)
    for i in range (0,iteration):
        ind = np.random.choice(size,size=1,replace = False,p=None)
        b = b - e*(2*lamda*b - ((y_train[ind]-(1/(1+np.exp(np.dot(x_train[ind],-b)))))*x_train[ind]).T)
        loss_sgd[i] = lamda*np.dot(b.T,b)-(np.log((1/(1+np.exp(np.dot(x_train[ind],-b)))))*y_train[ind])-(np.log(1-(1/(1+np.exp(np.dot(x_train[ind],-b)))))*(1-y_train[ind]))
    plt.plot([x for x in range(1,iteration+1)],loss_sgd)
    plt.xlabel("No. of Iterations")
    plt.ylabel("Trainning Loss")
    plt.title("Training Loss of SGD(Standardized data)")
    plt.show()
    return b

def LR_sgd_adapt(x_train,y_train,e,lamda,iteration):
    b0=np.sqrt(2.0/57)*np.random.randn(57)
    b = b0.reshape(57,1)
    size = x_train.shape[0]
    loss_sgd = np.zeros(iteration)
    for i in range (1,iteration):
        ind = np.random.choice(size,size=1,replace = False,p=None)
        b = b - (e/i)*(2*lamda*b - ((y_train[ind]-(1/(1+np.exp(np.dot(x_train[ind],-b)))))*x_train[ind]).T)
        loss_sgd[i] = lamda*np.dot(b.T,b)-(np.log((1/(1+np.exp(np.dot(x_train[ind],-b)))))*y_train[ind])-(np.log(1-(1/(1+np.exp(np.dot(x_train[ind],-b)))))*(1-y_train[ind]))

    plt.plot([x for x in range(1,iteration+1)],loss_sgd)
    plt.xlabel("No. of Iterations")
    plt.ylabel("Trainning Loss")
    plt.title("Training Loss of SGD with varied learning rate(Binary data)")
    plt.show()
    return b

def predict(x_train,y_train,model):
    b = model
    x_pred = 1/(1+np.exp(np.dot(x_train,-b)))
    x_final = np.zeros((x_pred.shape[0],x_pred.shape[1]))
    for i in range(0,x_pred.shape[0]):
        if x_pred[i]>0.5:
            x_final[i]=1
        else:
            x_final[i]=0
    hit = 0
    for i in range(0,x_pred.shape[0]):
        if x_final[i]==y_train[i]:
            hit = hit + 1
    acc = float(hit)/x_pred.shape[0]
    print "Training accuracy is %s (%s/%s)" %(acc,hit,x_pred.shape[0])

def predict_test(x_train,model):
    b = model
    x_pred = 1/(1+np.exp(np.dot(x_train,-b)))
    x_final = np.zeros((x_pred.shape[0],x_pred.shape[1]))
    for i in range(0,x_pred.shape[0]):
        if x_pred[i]>0.5:
            x_final[i]=1
        else:
            x_final[i]=0
    return x_final

if __name__ == "__main__":
    mat = scipy.io.loadmat('spam.mat')
    ytrain = mat['ytrain']
    xtrain = mat['Xtrain']
    xtest = mat['Xtest']
    xtrain_std = data_std(xtrain)
    xtest_std = data_std(xtest)
    xtrain_log = data_log(xtrain)
    xtrain_bin = data_bin(xtrain)
    
# For the following model training part, Please only enable one model training and disable the rest of them.
# In addition, you need to modify the model name to predict() function accordingly.
#================train with standardized data=========================#
    model = LR_gd(xtrain_std,ytrain,0.00001,0.000001,3000)
    model_sgd = LR_sgd(xtrain_std,ytrain,0.0001,5,10000)
#====================train with log data==============================#
    log_model = LR_gd(xtrain_log,ytrain,1e-8,0.0001,2000000)
    log_model_sgd = LR_sgd(xtrain_log,ytrain,1e-6,100,30000)
#==================train with Binary data=================================#
    bin_model = LR_gd(xtrain_bin,ytrain,0.00001,0.000001,20000)
    bin_model_sgd = LR_sgd(xtrain_bin,ytrain,1e-5,5,60000)
#===============SGD train with adaptive learning Rate=====================#
    model_sgd_adapt = LR_sgd_adapt(xtrain_std,ytrain,0.001,50,5000)
    log_model_sgd_adapt = LR_sgd_adapt(xtrain_log,ytrain,1e-3,100,3000)
    bin_model_sgd_adapt = LR_sgd_adapt(xtrain_bin,ytrain,1e-3,50,5000)
#================================Prediction=================================#      
    predict(xtrain_bin,ytrain,bin_model_sgd_adapt)
    
    
    
#================================Test output CSV==============================#    
    ytest = predict_test(xtest_std,bin_model)
    yfinal = np.zeros((ytest.shape[0],2))
    for i in range (0,ytest.shape[0]):
        yfinal[i][0]=int(i+1)
        yfinal[i][1]=int(ytest[i])
    
    with open('cxxx.csv', "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter = ',', lineterminator = '\r\n')
        for line in yfinal:
            output = line
            writer.writerow(output)
