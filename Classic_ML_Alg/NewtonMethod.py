import numpy as np
import matplotlib.pyplot as plt

def data_std(xtrain):
    return ((xtrain - np.mean(xtrain, axis=0)) / np.std(xtrain, axis=0))

xx = np.array([4,5,5.6,6.8,7,7.2,8,0.8,1,1.2,2.5,2.6,3,4.3])
nox = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
nox2=nox.reshape(14,1)
xxx = data_std(xx)
xxxx = xxx.reshape(14,1)
x = np.concatenate((xxxx, nox2),1)
x = np.append(x, np.array([[3,1]]), axis=0)
yy = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
y=yy.reshape(15,1)
la = 0.07
b = np.array([1,0])
b0 = b.reshape(2,1)
#========================Logistic regression with Newton=========================#
u0 = 1/(1+np.exp(np.dot(x,-b0)))
pl0=plt.scatter(x[:,0], u0,color='r',marker='+')


W = np.zeros((15, 15))
for i in range (15):
    W[i][i] = u0[i]*(1-u0[i])
    
x2 = np.linalg.inv(2*la*np.identity(2)+np.dot((np.dot(x.T,W)),x))

x1 =(2*la*b0-np.dot(x.T,y-u0))

b1 = b0-np.dot(x2,x1)
print b1

u1 = 1/(1+np.exp(np.dot(x,-b1)))
pl1=plt.scatter(x[:,0], u1,color='g',marker='*')

W1 = np.zeros((15, 15))
for i in range (15):
    W1[i][i] = u1[i]*(1-u1[i])
x22 = np.linalg.inv(2*la*np.identity(2)+np.dot((np.dot(x.T,W1)),x))

x12 =(2*la*b1-np.dot(x.T,y-u1))

b2 = b1-np.dot(x22,x12)
print b2
u2 = 1/(1+np.exp(np.dot(x,-b2)))
pl2=plt.scatter(x[:,0], u2,color='k',marker='s')

W2 = np.zeros((15, 15))
for i in range (15):
    W2[i][i] = u2[i]*(1-u2[i])
x23 = np.linalg.inv(2*la*np.identity(2)+np.dot((np.dot(x.T,W2)),x))

x13 =(2*la*b2-np.dot(x.T,y-u2))

b3 = b2-np.dot(x23,x13)
print b3
u3 = 1/(1+np.exp(np.dot(x,-b3)))
pl3 = plt.scatter(x[:,0], u3)

plt.legend((pl0, pl1, pl2,pl3),
           ('u0', 'u1', 'u2', 'u3'),
           scatterpoints=1,
           loc='upper left',
           ncol=4,
           fontsize=12)
plt.show()

#======================Linear regression with Newton============================#
line_x2 = np.linalg.inv(2*la*np.identity(2)+np.dot(x.T,x))

line_x1 =(2*la*b0-np.dot(x.T,y-np.dot(x,b0)))
plt.scatter(x[:,0], yy,color='r',marker='+')
b1 = b0-np.dot(line_x2,line_x1)
print b1
#b11=b1
#b11[0]=(-1/b1[0])
#decision boundary
#plt.plot(x[:,0],np.dot(x,b11),color='b')
plt.plot(x[:,0],np.dot(x,b1),color='b')
plt.show()

#================Logistic fit, Linear fit, Original data========================#
fig, ax = plt.subplots()
ax.scatter(x[:,0], yy,color='r',marker='+',label='Data Points')

t = np.arange(-1.5, 1.5, 0.1)
xplot = np.zeros((30,2))
for i in range(0,30):
    xplot[i][0]=t[i]
    xplot[i][1]=1
ax.plot(xplot[:,0], 1/(1+np.exp(np.dot(xplot,-b3))),'g--',label='Logistic fit')

ax.plot(x[:,0],np.dot(x,b1),color='b',label='Linear fit')
legend = ax.legend(loc='upper left', shadow=True)
plt.show()
