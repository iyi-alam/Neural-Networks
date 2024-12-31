import matplotlib.pyplot as plt
import numpy as np

LOOK_UP_LABELS = {
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

#Generate one random picture from each class and print actual and test values
def Format_Image_MNIST(image, RGB = 'False'):
    #I am assuming that image is a matrix with RGB channels stacked in columns
    #it will return 32x32x3 imshow friendly RGB scaled image
    #image = image/(np.max(image))
    image = image*255
    image = np.clip(image,0,255)
    image = image.reshape(28,28)
    image = image.astype(np.uint8)
    return image

def PlotImages_MNIST(images, actual_labels, predicted_labels):
    cols = images.shape[0]
    fig, axes = plt.subplots(1,cols, figsize = (20,3))
    for k in range(cols):
        axes[k].imshow(Format_Image_MNIST(images[k,:]), cmap='gray')
        axes[k].set_title(f"Original: {LOOK_UP_LABELS[actual_labels[k]]}\n"+ \
                            f"Predicted: {LOOK_UP_LABELS[predicted_labels[k]]}", fontsize = 10)
        axes[k].axis('off')
        
    plt.savefig("fashion_mnist_predicted.png")

def Create_Visualaization(X_test, Y_test, Y_pred):
    random_indices = []
    for row in range(Y_test.shape[0]):
        true_indices = np.where(Y_test[row,:]==1)[0]
        select_indices = np.random.choice(true_indices)
        random_indices.append(select_indices)
    sample_image = X_test[random_indices,:]
    actual_labels = np.argmax(Y_test[:,random_indices], axis = 0)

    predicted_label = np.argmax(Y_pred , axis = 0)

    PlotImages_MNIST(sample_image, actual_labels, predicted_label)
    return 


def one_hot_encode(ytrain):
        """
        Converts a label vector into a one-hot encoded matrix.
        """
        Y_train = np.squeeze(ytrain).astype(int)
        N = Y_train.shape[0]
        K = len(np.unique(Y_train))
        Y_one_hot = np.zeros((N, K))
        
        # Use numpy's advanced indexing to set the appropriate class column to 1 for each row
        Y_one_hot[np.arange(N), Y_train] = 1

        return Y_one_hot.T
    
def accuracy(_ypred: np.ndarray, _yactual: np.ndarray):
    ypred = np.argmax(_ypred, axis = 0)
    yactual = np.argmax(_yactual, axis = 0)
    N = _ypred.shape[1]
    acc = np.sum(ypred==yactual)
    return acc/N