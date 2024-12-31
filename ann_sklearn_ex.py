import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs,make_classification
import ann_model as ann
import Optimizers

#Construct different Datasets for Training and testing
def Gaussian_Mixture():
    x,y = make_blobs(
    n_samples=4000,     # Number of points
    n_features=5,       # Number of dimensions (2 for easy visualization)
    centers=3,          # Number of classes
    cluster_std=1.5,    # Spread of each cluster
    random_state=42
    )
    dataset = np.hstack((x,y.reshape(-1,1)))
    return dataset

def Sklearn_Classification():
    # Generate synthetic dataset
    x, y = make_classification(
        n_samples=10000,     # Number of examples
        n_features=10,      # Number of features
        n_classes=5,        # Number of classes
        n_informative=10,   # Number of informative features
        n_redundant=0,      # Number of redundant features
        random_state=42
    )
    dataset = np.hstack((x,y.reshape(-1,1)))
    return dataset

def trainTestSplit(X,train_size):
    N = X.shape[0]
    indices = np.array(range(N))
    np.random.shuffle(indices)
    X_train = X[indices[:train_size],:]
    X_test = X[indices[train_size:],:]
    return X_train, X_test

dataset = Gaussian_Mixture()
N_data,N_features = dataset.shape[0], (dataset.shape[1]-1)
train_size = int(0.75*N_data)
test_size = N_data - train_size
train_data, test_data = trainTestSplit(dataset, train_size)
 
X_train = train_data[:, :N_features]
Y_train = train_data[:,N_features:].astype(int)
#print(X_train.shape, Y_train.shape)

X_test = test_data[:, :N_features]
Y_test = test_data[:,N_features:].astype(int)
#print(X_test.shape, Y_test.shape)

N_labels = len(np.unique(Y_train))

Y_train = ann.Helper_functions.one_hot_encode(Y_train)
Y_test = ann.Helper_functions.one_hot_encode(Y_test)

#Create a neural network model
max_epochs = 30
batch_size = 64
neurons = (64,32, N_labels)
activations = (ann.ReLU, ann.ReLU, ann.softmax)
model = ann.MLLclassification(in_features= N_features,
                          out_features=N_labels, neurons = neurons, 
                          activations = activations)

#select optimizer and loss function
#optimizer = Optimizers.Momentum(lr = 0.01, beta =0.9)
optimizer = Optimizers.Adam(lr = 0.01, beta1 = 0.9, beta2 = 0.99, epsilon= 1e-8)
lossfunc = ann.cross_entropy_loss()

#set up ann optimization module
optimizer_class = ann.Optimization(model, optimizer= optimizer)

#Create training loop 
for epoch in range(max_epochs):

    num_batches = int(X_train.shape[0]/batch_size) #Number of batches to be processed

    for batch in range(num_batches):

        st_index = batch*batch_size
        end_index = (batch + 1)*batch_size
        if (end_index > X_train.shape[0]):
            end_index = X_train.shape[0]
            st_index = end_index - batch_size
        
        x = X_train[st_index:end_index,:]
        ytarget = Y_train[:,st_index:end_index]
        ypred = model.forward(x.T)
        loss = lossfunc(ytarget, ypred) #normalized batch loss
        optimizer_class.backward_pass(ytarget= ytarget, ypred= ypred, lossfunc = lossfunc)
        optimizer_class.update_params()
    
    #Compute Normalized Train and Test Loss
    y_train_predict = model.forward(X_train.T) 
    y_test_predict = model.forward(X_test.T)

    train_loss = lossfunc(Y_train, y_train_predict)/Y_train.shape[1]
    test_loss = lossfunc(Y_test, y_test_predict )/Y_test.shape[1]

    train_acc = ann.Helper_functions.accuracy(y_train_predict, Y_train)
    test_acc = ann.Helper_functions.accuracy(y_test_predict, Y_test)
    
    if epoch % 10 == 0 or epoch == max_epochs - 1:
        print(f"Epoch {epoch + 1}/{max_epochs} - Train Loss: {train_loss:.2f}\n"  
              f"Train Accuracy: {train_acc:.2}\tValidation Accuracy: {test_acc:.2}\n")

#Test the accuracy of the network
ypred = model.forward(X_test.T)
acc = ann.Helper_functions.accuracy(ypred, Y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")


