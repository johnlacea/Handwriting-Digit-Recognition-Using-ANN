"""
    Programmer  : JOHN B. LACEA
    Place       : Bilis, Burgos, La Union 2510
    Date        : June 12, 2022
    Description : Digit Classification on the MNIST dataset
                  I modified and improved the code of Samson Zhang to predict digits
                  from test data and images instead from the trained datasets.
                  Building Neural Networks from scratch - No TensorFlow
                  It uses numpy, pandas, matplotlib, dill and imageio libraries.
                  
                  +++ It stores the W1, b1, W2, and b2 objects for fast testing for prediction! +++
                  
                  Hyper-Parameters:
                  input layer nodes is 784 because 28x28 = 784 pixels
                  hidden layer nodes is 10
                  output layer nodes is 10 because there are 10 digits (0 to 9)
                  
                  MNIST Dataset
                  Pixel 0 = means background WHITE
                  Pixel 1..255 = means foreground BLACK
                  
                  Test #1
                  Training Data is 60,000 handwriting records
                  Predicting 2.png is 2
                  Predicting 3.png is 3
                  Predicting 5.png is 5
                  Predicting 8.png is 5
                  
                  Accuracy is 3/4 = 75%
                  Test error rate is 1/4 = 25%
                  
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
                  4. pip install pandas
                  5. pip install imageio
                  6. python -m pip install -U matplotlib [--prefer-binary]
                  
                  You are hereby to use this code for free. God Bless Us!
"""
import dill as pickle
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

print("+++ Loading Training Data +++")
data = pd.read_csv('mnist_dataset/mnist_train_60K.csv',header=None)

data = np.array(data)
m, n = data.shape   #m is total rows
print("m is {:d} and n is {:d}".format(m,n))
#The data is shuffled because the order of the data
#can cause the network to become biased during the Training of the Neural Network model.
np.random.shuffle(data) # shuffle before splitting into training sets

#Training data
data_train = data[0:m].T
print("data_train shape is {}".format(data_train.shape))
Y_train = data_train[0]    #Validation data-These are digit-labels as key answers. It is used to minimize a problem known as overfitting.
X_train = data_train[1:n]  #Training data
#Scale X_Train data from 0 to 1
X_train = X_train / 255.   #Scale the Training data from 0 to 1
_,m_train = X_train.shape
print("X_train shape is {}".format(X_train.shape))
print("Y_train shape is {}".format(Y_train.shape))
print("Key Labels are {}".format(Y_train))

#It initializes the weights and biases randomly!
def init_params():
    W1 = np.random.rand(10, 784) - 0.5 # 10x784 2D array of weights at 1st layer is hidden layer
    b1 = np.random.rand(10, 1) - 0.5   # 10x1 2D array of bias at 1st layer is hidden layer
    W2 = np.random.rand(10, 10) - 0.5  # 10x10 2D array of weights at 2nd layer is output layer
    b2 = np.random.rand(10, 1) - 0.5   # 10x1 2D array of bias at 2nd layer is output layer
    return W1, b1, W2, b2

# Rectified Linear Unit(ReLU) is an activation function
def ReLU(Z):
    return np.maximum(Z, 0)

# Softmax is an activation function
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# ForwardPropagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1   #Gradient/slope for the 1st layer is hidden layer
    A1 = ReLU(Z1)         #1st layer is hidden layer
    Z2 = W2.dot(A1) + b2  #Gradient/slope for the 2nd layer is output layer
    A2 = softmax(Z2)      #2nd layer is output layer
    return Z1, A1, Z2, A2

#Determines the sign of the Gradient/Slope Z
def ReLU_deriv(Z):
    return Z > 0

#It is a one-hot encoding for the Y_train validation data that
#create new 10 classifications that take on values of 0 and 1
#to represent the original 10 classifications of digit values.
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# BackwardPropagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

#Adjusts the weights and biases after the backwardpropagation during the gradient descent!
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

#It returns the predicted digit from the neural network model!
def get_predictions(A2):
    return np.argmax(A2, 0)

#It computes the average of correct predictions of the neural network model
#by counting the correct predicted digit and divides to the total items.
def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

#Gradient Descent is an optimization
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration {} out of {}".format(i,iterations))
            predictions = get_predictions(A2)            
            print("Accuracy is %.2f%%" % (get_accuracy(predictions, Y) * 100))
    return W1, b1, W2, b2

# Train the Neural Network
print ("+++ Start Train the Neural Network +++")    
#Get the Training Results from Gradient Descent
alpha = 0.10 # Learning Rate is a hyper-parameter
iterations = 1500 #iterations is a hyper-parameter
print("+++ Gradient Descent %d iterations alpha = %.2f%% +++" % (iterations, alpha * 100))
start = time.time()
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, iterations)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print ("+++ End Train the Neural Network +++")
print("{:0>1} hour(s) {:0>1} minute(s) {:0>1} second(s)".format(int(hours),int(minutes),int(seconds)))

#Save the W1, b1, W2, b2 to pickel files
print ("+++ Save the W1, b1, W2, b2 to pickel files +++")
with open('pickle/W1.pkl','wb') as f:
  pickle.dump(W1,f,pickle.HIGHEST_PROTOCOL)
with open('pickle/b1.pkl','wb') as f:
  pickle.dump(b1,f,pickle.HIGHEST_PROTOCOL)
with open('pickle/W2.pkl','wb') as f:
  pickle.dump(W2,f,pickle.HIGHEST_PROTOCOL)
with open('pickle/b2.pkl','wb') as f:
  pickle.dump(b2,f,pickle.HIGHEST_PROTOCOL)

#The Neural Network model predicts the digit from the 784 pixels data.
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

#It opens a 28x28 image file and performs the digit prediction using the
#final computed weights and biases.
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

print("+++ Loading Test Data +++")
data = pd.read_csv('mnist_dataset/mnist_test_10K.csv',header=None)

data = np.array(data)
m, n = data.shape   #m is total rows and n is total columns
print("m is {:d} and n is {:d}".format(m,n))

#Test data
data_test = data[0:m].T
Y_test = data_test[0]    #Validation data-These are digit-labels as key answers.
X_test = data_test[1:n]  #Test data
#Scale X_test data from 0 to 1
X_test = X_test / 255.   #Scale the Test data from 0 to 1
X_test = X_test.T
print("X_test shape is {}".format(X_test.shape))
print("Y_test shape is {}".format(Y_test.shape))
print("Key Labels are {}".format(Y_test))

print("+++ Test Trained Neural Network on Test Data +++")
# Test the Neural Network
# scorecard for how well the network performs, initially empty
scorecard = []

for i in range(0,m):
    record = X_test[i].reshape(784,1)    
    prediction = make_predictions(record, W1, b1, W2, b2)    
    correct_label = Y_test[i]
    if (prediction[0] == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)        

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print("Score is {}/{}".format(scorecard_array.sum(),scorecard_array.size))
print ("performance = {:.2f}%".format((scorecard_array.sum() / scorecard_array.size) * 100.0))

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