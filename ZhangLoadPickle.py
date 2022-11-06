"""
    Programmer  : JOHN B. LACEA
    Place       : Bilis, Burgos, La Union 2510
    Date        : June 12, 2022
    Description : Digit Classification on the MNIST dataset
                  I modified and improved the code of Samson Zhang to predict digits
                  from test data and images instead from the trained datasets.
                  Building Neural Networks from scratch - No TensorFlow
                  It uses numpy, matplotlib, dill and imageio libraries.
                  
                  +++ It loads the stored W1, b1, W2, and b2 objects for prediction! +++
                                    
                  MNIST Dataset
                  Pixel 0 = means background WHITE
                  Pixel 1..255 = means foreground BLACK
                  
                  Test #1
                  Training Data is 60,000 handwriting records
                  Predicting 2.png is 2
                  Predicting 3.png is 3
                  Predicting 5.png is 2
                  Predicting 8.png is 3
                  
                  Accuracy is 2/4 = 50%
                  Test error rate is 2/4 = 50%
                  
                  Test #2
                  Training Data is 42,000 handwriting records
                  Predicting 2.png is 2
                  Predicting 3.png is 3
                  Predicting 5.png is 2
                  Predicting 8.png is 5
                  
                  Accuracy is 2/4 = 50%
                  Test error rate is 2/4 = 50%
                  
                  Needs to install the following libraries:
                  1. python -m pip install -U pip  # Update the pip package manager
                  2. pip install dill
                  3. pip install numpy
                  4. pip install imageio
                  5. python -m pip install -U matplotlib [--prefer-binary]
                  
                  You are hereby to use this code for free. God Bless Us!
"""
import dill as pickle
import imageio
import numpy as np
from matplotlib import pyplot as plt

try:
    #Load pickle files
    print ("+++ Load pickle files namely: W1, b1, W2, b2 +++")
    with open('pickle/W1.pkl','rb') as f:
      W1 = pickle.load(f)
    with open('pickle/b1.pkl','rb') as f:
      b1 = pickle.load(f)
    with open('pickle/W2.pkl','rb') as f:
      W2 = pickle.load(f)
    with open('pickle/b2.pkl','rb') as f:
      b2 = pickle.load(f)
except Exception as e:
    print("Error: %r" % (e))
    exit()
  
# Rectified Linear Unit(ReLU) is an activation function
def ReLU(Z):
    return np.maximum(Z, 0)

# Softmax is an activation function
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1  #1st layer is hidden layer
    A1 = ReLU(Z1)        #1st layer is hidden layer
    Z2 = W2.dot(A1) + b2  #2nd layer is output layer
    A2 = softmax(Z2)      #2nd layer is output layer
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def Prediction(file, W1, b1, W2, b2):
    print("Predicting the file: ", file)
    img_array = imageio.v2.imread(file, as_gray=True)
        
    # reshape from 28x28 to list of 784 values, invert values (switch 0 to WHITE as foreground color and 1..255 to BLACK as background color obeying the MNIST pixel format)
    img_data  = 255.0 - img_array.reshape(784)
        
    # then scale data to range from 0 to 1
    img_data = img_data / 255.0
        
    arr = img_data.reshape(784,1)    
    prediction = make_predictions(arr, W1, b1, W2, b2)
    print("Predicted digit is ", prediction[0])
        
    # plot image following the MNIST pixel format    
    plt.imshow(img_data.reshape(28,28), cmap='gray', interpolation='None')
    plt.show()

print("+++ Test Trained Neural Network on Example Images +++")
#Predicting the digit from images
Prediction('images/2.png', W1, b1, W2, b2)
Prediction('images/3.png', W1, b1, W2, b2)
Prediction('images/5.png', W1, b1, W2, b2)
Prediction('images/8.png', W1, b1, W2, b2)

print("\nPress any key to exit!")    
#Pause for Sceen View
input()
exit()