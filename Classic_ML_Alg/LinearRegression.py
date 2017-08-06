from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import math

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    xtx = X_train.T.dot(X_train)
    return scipy.linalg.solve(xtx + reg*np.eye(xtx.shape[0]), X_train.T.dot(y_train), sym_pos=True)

def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    xTrans = X_train.transpose() #5000*60000
    m=X_train.shape[0]
    theta = np.ones((X_train.shape[1],y_train.shape[1]))
    for i in range(0, num_iter):
        guess = np.dot(X_train, theta)
        loss = guess - y_train
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=100000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    m=X_train.shape[0]
    theta = np.ones((X_train.shape[1],y_train.shape[1]))
    for i in range(0, num_iter):
        j = i%m
        guess = np.dot(X_train[j], theta)
        loss = guess - y_train[j]
        losstran=np.reshape(loss,(1,10))
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        xTrans=X_train[j].transpose()
        xTrans1=np.reshape(xTrans,(5000,1))
        gradient = np.dot(xTrans1, losstran) / m
        theta = theta - alpha * gradient
    return theta

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.eye(NUM_CLASSES)[labels_train]

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(X.dot(model), axis=1)

def phi(X):
    ''' Featurize the inputs using random Fourier features '''
    mu, sigma = 0, 5
    G = np.random.normal(mu, sigma, (X.shape[1],5000))
    b = np.random.uniform(0, 2*math.pi,5000)
    X1 = np.dot(X,G)
    for i in range(X1.shape[0]):
        X1[i]=np.cos(X1[i]+b)
    return X1


if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    X_train, X_test = phi(X_train), phi(X_test)

    model = train(X_train, y_train, reg=0.1)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Closed form solution")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    model = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=50000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    model = train_sgd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=1000000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Stochastic gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

