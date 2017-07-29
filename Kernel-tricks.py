#This project is used for leveraging polynomial and gaussian kernel to predict label.

import numpy as np
import math
import matplotlib.pyplot as plt

def poly_kernel(x,y):
    K = np.zeros((x.shape[0],y.shape[0]))
    for i in range(x.shape[0]):
        xi = x[i,:]
        for j in range(y.shape[0]):
            xj = y[j,:]
            K[i,j] = (1+np.dot(xi,xj.T))**2
    return K

def gaussian_kernel(x,y,sigma):
    K = np.zeros((x.shape[0],y.shape[0]))
    for i in range(x.shape[0]):
        xi = x[i,:]
        for j in range(y.shape[0]):
            xj = y[j,:]
            K[i,j]=math.exp(-sigma*np.dot((xi-xj).T,(xi-xj)))
    return K

def solve(k,y,lamda):
    I = np.eye(y.shape[0])
    a = np.dot(np.linalg.inv(k+lamda*I),y)
    return a

def predict(a,k,y):
    hit = 0
    h = np.dot(k,a)
    pre = np.zeros_like(h)
    for j in range(200):
        if h[j]>=0:
            pre[j]=1
        elif h[j]<0:
            pre[j]=-1
    for i in range (200):
        if pre[i]==y[i]:
            hit += 1
    print "accuracy is %s (%s/%s)" %(float(hit)/200,hit, 200)
    return pre
    

    
theta = np.random.uniform(0,2*math.pi,100)
w1=np.random.normal(0,1,100)
w2=np.random.normal(0,1,100)
v1=np.random.normal(0,1,100)
v2=np.random.normal(0,1,100)

x=np.zeros((200,2))

x1=np.zeros((100,2))
x2=np.zeros((100,2))
for m in range (100):
    x[m]=(8*math.cos(theta[m])+w1[m],8*math.sin(theta[m])+w2[m])
    x1[m]=(8*math.cos(theta[m])+w1[m],8*math.sin(theta[m])+w2[m])

for n in range (100,200):
    x[n]=(v1[n-100],v2[n-100])
    x2[n-100]=(v1[n-100],v2[n-100])

y=np.ones((200,1))
for i in range (100,200):
    y[i] = -1

p_ker = poly_kernel(x,x)
a=solve(p_ker,y,1e-6)
pre = predict (a,p_ker,y)
print "================================"
g_ker = gaussian_kernel(x,x,0.001)
a_gau = solve(g_ker,y,1e-4)
pre2 = predict (a_gau,g_ker,y)
#You can enable either polynomial contour plot, or gaussian contour plot once at a time
'''
#============polynomial contour plot===============#
scale=0.5
x_min, x_max = x[:, 0].min()-1, x[:, 0].max()+1
y_min, y_max = x[:, 1].min()-1, x[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max,scale),
                     np.arange(y_min, y_max,scale))

c1 = xx.ravel()
c1 = c1.reshape(c1.shape[0],1)
c2 = yy.ravel()
c2 = c2.reshape(c2.shape[0],1)

xxx= np.concatenate((c1,c2),axis=1)

k_plot = poly_kernel(xxx,x)

h = np.dot(k_plot,a)

h = h.reshape(xx.shape)

plt.contourf(xx, yy, h,cmap=plt.cm.gray_r)'''
#============gaussian contour plot===============# this is used for ploting the figures
scale=0.5
x_min, x_max = x[:, 0].min()-1, x[:, 0].max()+1
y_min, y_max = x[:, 1].min()-1, x[:, 1].max()+1
xx_gau, yy_gau = np.meshgrid(np.arange(x_min, x_max,scale),
                     np.arange(y_min, y_max,scale))

c1 = xx_gau.ravel()
c1 = c1.reshape(c1.shape[0],1)
c2 = yy_gau.ravel()
c2 = c2.reshape(c2.shape[0],1)

xxx_gau= np.concatenate((c1,c2),axis=1)

k_gaussian_plot = gaussian_kernel(xxx_gau,x,0.001)

h_gau = np.dot(k_gaussian_plot,a_gau)

h_gau = h_gau.reshape(xx_gau.shape)

plt.contourf(xx_gau, yy_gau, h_gau,cmap=plt.cm.gray_r)
#============data point scattr plot===============#
p1=plt.scatter(x1[:,0],x1[:,1],marker='o',color='r')
p2=plt.scatter(x2[:,0],x2[:,1],marker='^',color='b')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((p1,p2),('y=1','y=-1'))
plt.show()


