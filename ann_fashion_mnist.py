import numpy as np
import matplotlib.pyplot as plt
import ann_model as ann
import Optimizers
import utils
import os

#Load the MNIST digits dataset from the local folder
print("\nLoading the MNIST dataset... \n")
# data_folder = "D:\\Linear Algebra\\Assignments\\Programming\\"

# train_file = os.path.join(data_folder, "fashion-mnist_train.csv")
# test_file = os.path.join(data_folder, "fashion-mnist_test.csv")

train_data = np.loadtxt("fashion-mnist_train.csv", delimiter = ",", skiprows=1)
test_data = np.loadtxt("fashion-mnist_test.csv", delimiter = ",",skiprows=1)

train_images, train_labels = train_data[:,1:], train_data[:,0].reshape((-1,1))
test_images, test_labels = test_data[:,1:], test_data[:,0].reshape((-1,1))

#Prepare the dataset as per the requirement of the model
N_data = train_images.shape[0]                      #Number of images in training data
N_features = train_images.shape[1]                  #Number of features
N_labels = len(np.unique(train_labels))             #Number of unique class labels
# Lookup dictionary for MNIST Fashion dataset
fashion_mnist_lookup = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


print("\nPreparing the MNIST dataset...\n")
validation_set_size = int(0.1*N_data)
X_train = train_images[:(N_data-validation_set_size), :]/255
Y_train = train_labels[:(N_data-validation_set_size), :]

X_validation = train_images[(N_data-validation_set_size):, :]/255
Y_validation = train_labels[(N_data-validation_set_size):, :]

X_test = test_images/255
Y_test = test_labels

print("Shape of Xtrain and Ytrain\n")
print(X_train.shape, Y_train.shape)

print("Shape of Xvalidate and Yvalidate\n")
print(X_validation.shape, Y_validation.shape)

print("Shape of Xtest and Ytest:\n")
print(X_test.shape, Y_test.shape)


#Prepare the one hot encoded output
Y_train = utils.one_hot_encode(Y_train)
Y_validation = utils.one_hot_encode(Y_validation)
Y_test = utils.one_hot_encode(Y_test)

print("Shape of One-hot encoded output:\n")
print(Y_train.shape, Y_validation.shape , Y_test.shape)

#Create a neural network model
max_epochs = 20
batch_size = 64
neurons = (64, 32,16, N_labels)
activations = (ann.ReLU, ann.ReLU, ann.ReLU, ann.softmax)
model = ann.MLLclassification(in_features= N_features,
                          out_features=N_labels, neurons = neurons, 
                          activations = activations)

#select optimizer and loss function
optimizer = Optimizers.Adam(lr = 0.01, beta1 = 0.9, beta2 = 0.99, epsilon= 1e-8)
lossfunc = ann.cross_entropy_loss()

#set up ann optimization module
optimizer_class = ann.Optimization(model, optimizer= optimizer)

# Set the training to true if you want to track the training
TRAIN_MODE = True
print("Training the model:\n")
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
    
    if TRAIN_MODE:
        #Track training for hyperparameter tuning
        #Compute Normalized Train and Test Loss
        y_train_predict = model.forward(X_train.T) 
        y_validation_predict = model.forward(X_validation.T)

        train_loss = lossfunc(Y_train, y_train_predict)/Y_train.shape[1]
        validation_loss = lossfunc(Y_validation, y_validation_predict)/Y_validation.shape[1]

        train_acc = utils.accuracy(y_train_predict, Y_train)
        validation_acc = utils.accuracy(y_validation_predict, Y_validation)
        
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch + 1}/{max_epochs} - Train Loss: {loss:.2f}\n"  
                f"Train Accuracy: {train_acc:.2}\tValidation Accuracy: {validation_acc:.2}\n")

#Test the accuracy of the network
ypred = model.forward(X_test.T)
acc = utils.accuracy(ypred, Y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")


# Create visualization to see the actual and predicted labels
utils.Create_Visualaization(X_test, Y_test, ypred)


