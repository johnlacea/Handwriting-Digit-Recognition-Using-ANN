"""
    Programmer  : JOHN B. LACEA
    Place       : Bilis, Burgos, La Union 2510
    Date        : June 1, 2022
    Description : Digit Classification on the MNIST dataset
                  Make Your Own Neural Network Book by Tariq Rashid
                  It uses numpy, scipy, matplotlib, dill and imageio libraries.
                  
                  +++ It stores the object named n for fast testing for prediction! +++
                  
                  Hyper-Parameters:
                  input layer nodes is 784 because 28x28 = 784 pixels
                  hidden layer nodes is 200
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
                  Predicting 2.png is 7
                  Predicting 3.png is 3
                  Predicting 5.png is 5
                  Predicting 8.png is 8
                  
                  Accuracy is 2/4 = 50%
                  Test error rate is 2/4 = 50%
                  
                  Needs to install the following libraries:
                  1. python -m pip install -U pip  # Update the pip package manager
                  2. pip install dill
                  3. pip install imageio
                  4. pip install numpy
                  5. pip install scipy
                  6. python -m pip install -U matplotlib [--prefer-binary]
                  
                  You are hereby to use this code for free. God Bless Us!
                  
"""
import dill as pickle
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# helper to load data from PNG image files
import imageio
import os
import time

#Global Variables
# number of input, hidden and output nodes are hyper-parameters
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# Learning Rate is a hyper-parameter
learning_rate = 0.1

# neural network class definition
class NeuralNetwork:    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)        
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))        
    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

def Prediction(file,n):
    # load image data from png files into an array
    print("Predicting the file: ", file)
    img_array = imageio.v2.imread(file, as_gray=True)
        
    # reshape from 28x28 to list of 784 values, invert values (switch 0 to WHITE as foreground color and 1..255 to BLACK as background color obeying the MNIST pixel format)
    img_data  = 255.0 - img_array.reshape(784)
    
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print("min = ", numpy.min(img_data))
    print("max = ", numpy.max(img_data))
    
    # query the neural network
    outputs = n.query(img_data)
    print (outputs)

    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("neural network says ", label)
    
    # plot image following the MNIST pixel format
    matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='gray', interpolation='None')
    matplotlib.pyplot.show()
 
#MAIN
#os.system('cls')

# create instance of neural network
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)       
    
print("+++ Loading Training Data +++")
# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train_60K.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Train the Neural Network
print ("+++ Start Train the Neural Network +++")    
# epochs is the number of times the training data set is used for training
epochs = 10 #Epoch is a hyper-parameter
print ("epochs: {:d}".format(epochs))

start = time.time()
for e in range(epochs):
    print ("Train the Neural Network: epoch:{:d} out of {:d}".format(e+1,epochs))
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')                
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:],dtype='float') / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99 #Validation data is used to minimize a problem known as overfitting.
        n.train(inputs, targets)
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print ("+++ End Train the Neural Network +++")
print("{:0>1} hour(s) {:0>1} minute(s) {:0>1} second(s)".format(int(hours),int(minutes),int(seconds)))

#Save the object named n to pickel file
print ("+++ Save the object named n to pickel file +++")
with open('pickle/n.pkl','wb') as f:
  pickle.dump(n,f,pickle.HIGHEST_PROTOCOL)

print("+++ Loading Test Data +++")
# load the mnist test data CSV file into a list
test_data_file = open("mnist_dataset/mnist_test_10K.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

print("+++ Test Trained Neural Network on Test Data +++")
# Test the Neural Network
# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the neural network
    outputs = n.query(inputs)        

    # the index of the highest value corresponds to the label
    predicted_label = numpy.argmax(outputs)
    
    # append correct or incorrect to list
    # need to extract from pytorch tensor via numpy to compare to python integer
    if (predicted_label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)        

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("Score is {}/{}".format(scorecard_array.sum(),scorecard_array.size))
print ("performance = {:.2f}%".format((scorecard_array.sum() / scorecard_array.size) * 100.0))
                
print("+++ Test Trained Neural Network on Example Images +++")
# Predicting the digit from images
Prediction('images/2.png', n)
Prediction('images/3.png', n)
Prediction('images/5.png', n)
Prediction('images/8.png', n)

print("\nPress any key to exit!")    
#Pause for Sceen View
input()
exit()