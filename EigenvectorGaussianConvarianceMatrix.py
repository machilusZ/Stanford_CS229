#Decision tree and random forest for fast decision making

import scipy.io
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import csv

def divideset(rows,column,value):
    split_function=None
    if isinstance(value,int) or isinstance(value,float):
        split_function=lambda row:row[column]>=value
    else:
        split_function=lambda row:row[column]==value

    set1=[row for row in rows if split_function(row)]
    set2=[row for row in rows if not split_function(row)]
    return (set1,set2)

def uniquecounts(rows):
    results={}
    for row in rows:
#result is in the last column
        r=row[len(row)-1]
        if r not in results: results[r]=0
        results[r]+=1
    return results

def entropy(rows):
    from math import log
    log2=lambda x:log(x)/log(2)  
    results=uniquecounts(rows)
    ent=0.0
    for r in results.keys():
        p=float(results[r])/len(rows)
        ent=ent-p*log2(p)
    return ent

def decisiontree(rows):
    if len(rows) == 0: return decisionnode()
    current_score = entropy(rows)
    
    best_gain = 0.0
    best_criteria=None
    best_sets=None
    
    column_count = len(rows[0])-1 # last column is the target attribute
    y_total=uniquecounts(rows)
    
    for col in range(0,column_count):
        column_values={}
        for row in rows:
            column_values[row[col]]=1
        
        for value in column_values.keys():
            (set_a,set_b)=divideset(rows,col,value)
            p=float(len(set_a))/len(rows)
            gain=current_score-p*entropy(set_a)-(1-p)*entropy(set_b)
            if gain>best_gain and len(set_a)>0 and len(set_b)>0:
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set_a,set_b)
    
    if best_gain>0:
        true_branch=decisiontree(best_sets[0])
        false_branch=decisiontree(best_sets[1])
        return decisionnode(col=best_criteria[0],value=best_criteria[1],tb=true_branch,fb=false_branch)
    else:
        #return decisionnode(results=uniquecounts(rows))
        
        classes = uniquecounts(rows)
        classes_pre = map(int,classes.keys())
        if len(classes_pre)==1:
            return decisionnode(results=classes)
        else:
            if float(classes[0])/y_total[0]>classes[1]/y_total[1]:
                del classes[1]
                return decisionnode(results=classes)
            else:
                del classes[0]
                return decisionnode(results=classes)
            

def printtree(tree,indent=''):
    if tree.results!=None:
        print(str(tree.results))
    else:
        print('Column No. = '+str(tree.col)+' , '+'Column Value > '+str(tree.value)+'? ')
        print indent+'True->',
        printtree(tree.tb,indent+'  ')
        print indent+'False->',
        printtree(tree.fb,indent+'  ')

class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb


def classify(observation,tree):
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
        branch=None
        if isinstance(v,int) or isinstance(v,float):
            if v>=tree.value: branch=tree.tb
            else: branch=tree.fb
        else:
            if v==tree.value: branch=tree.tb
            else: branch=tree.fb
    return classify(observation,branch)

def predict_acc(xdata,ydata,tree):
    hit =0
    y_total=uniquecounts(xdata)
    for i in range (xdata.shape[0]):
        pre = classify(xdata[i],tree)
        x_pre = map(int,pre.keys())
        if len(pre)==1:
            if x_pre==ydata[i]:
                hit+=1
        else:
            #label is binary, so len(pre)==2
            current_pre=-1
            if float(x_pre[0])/y_total[0] > float(x_pre[1])/y_total[1]:
                current_pre = 0
                if current_pre==ydata[i]:
                    hit+=1
            else:
                current_pre = 1
                if current_pre==ydata[i]:
                    hit+=1        
    print "accuracy is %s (%s/%s)" %(float(hit)/xdata.shape[0],hit,xdata.shape[0])
    
def predict(xtest,tree):
    fake_y = np.zeros((xtest.shape[0],1))
    test = np.concatenate((xtest,fake_y),axis=1)
    y_total=uniquecounts(test)
    predict_values=np.zeros((xtest.shape[0],2))
    for i in range(xtest.shape[0]):
        pre_test = classify(test[i],tree)
        #print pre_test
        x_pre_test = map(int,pre_test.keys())
        #print x_pre_test
        predict_values[i][0]=i+1
        predict_values[i][1]=int(x_pre_test[0])

    np.savetxt("pred.csv",predict_values,fmt='%d',delimiter=",")
    
def RF_predict_acc(train,t1,t2,t3,t4,t5):
    hit = 0
    for i in range(train.shape[0]):
        pre1=classify(train[i],t1)
        x_pre1=map(int,pre1.keys())
        pre2=classify(train[i],t2)
        x_pre2=map(int,pre2.keys())
        pre3=classify(train[i],t3)
        x_pre3=map(int,pre3.keys())
        pre4=classify(train[i],t4)
        x_pre4=map(int,pre4.keys())
        pre5=classify(train[i],t5)
        x_pre5=map(int,pre5.keys())
        
        if x_pre1+x_pre2+x_pre3+x_pre4+x_pre5>4:
            x_pre=1
            if x_pre == train[i,32]:
                hit += 1
        else:
            x_pre=0
            if x_pre == train[i,32]:
                hit += 1
    
    print "accuracy is %s (%s/%s)" %(float(hit)/train.shape[0],hit,train.shape[0])
        
    
if __name__ == "__main__":
    mat = scipy.io.loadmat('./spam_data/spam_data.mat')
    ytrain = mat['training_labels']
    xtrain = mat['training_data']
    xtest = mat['test_data']
    train = np.concatenate((xtrain,ytrain.T),axis=1)
    #np.random.shuffle(train)
    tree=decisiontree(train)
    printtree(tree)

    predict_acc(train,ytrain.T,tree)
    predict(xtest,tree)
#================Random Forests================#
'''
    np.random.shuffle(train)
    x1,x2,x3,x4,x5=np.array_split(train,5)
    tree1=decisiontree(x1)
    printtree(tree1)
    print "==========================================="
    tree2=decisiontree(x2)
    printtree(tree2)
    print "==========================================="
    tree3=decisiontree(x3)
    printtree(tree3)
    print "==========================================="
    tree4=decisiontree(x4)
    printtree(tree4)
    print "==========================================="
    tree5=decisiontree(x5)
    printtree(tree5)
    print "==========================================="
    #RF_predict_acc(train,tree1,tree2,tree3,tree4,tree5)


#================census data===================#
with open('train_data.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    imp = Imputer(missing_values='?', strategy='mean', axis=0)
    imp.transform(reader)
    vec = DictVectorizer()
    xxx = vec.fit_transform(reader).toarray()
    #le = preprocessing.LabelEncoder()
    #le.fit(reader)
    #print vec.shape

tree=decisiontree(xxx)
#predict_acc(xxx,tree)
#predict(xtest,tree)

np.random.shuffle(train)
    x1,x2,x3,x4,x5=np.array_split(xxx,5)
    tree1=decisiontree(x1)
    printtree(tree1)
    print "==========================================="
    tree2=decisiontree(x2)
    printtree(tree2)
    print "==========================================="
    tree3=decisiontree(x3)
    printtree(tree3)
    print "==========================================="
    tree4=decisiontree(x4)
    printtree(tree4)
    print "==========================================="
    tree5=decisiontree(x5)
    printtree(tree5)
    print "==========================================="
    #RF_predict_acc(train,tree1,tree2,tree3,tree4,tree5)
'''
    

