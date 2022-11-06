"""
    Programmer  : JOHN B. LACEA
    Place       : Bilis, Burgos, La Union 2510
    Date        : June 1, 2022
    Description : Digit Classification on the MNIST dataset
                  Make Your Own Neural Network Book by Tariq Rashid
                  It uses numpy, matplotlib, dill and imageio libraries.
                  
                  +++ It loads the stored object named n for prediction! +++
                                    
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
                  5. python -m pip install -U matplotlib [--prefer-binary]
                  
                  You are hereby to use this code for free. God Bless Us!
                  
"""
import dill as pickle
import numpy
import matplotlib.pyplot
import imageio
import os

def Prediction(file,n):
    # load image data from png files into an array
    print("Predicting the file: ", file)
    img_array = imageio.v2.imread(file, as_gray=True)
        
    # reshape from 28x28 to list of 784 values, invert values (switch 0 to WHITE as foreground color and 1..255 to BLACK as background color obeying the MNIST pixel format)
    img_data  = 255.0 - img_array.reshape(784)
    
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    
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
def main():           
    #os.system('cls')    
    
    try:
        #Load pickle file
        print ("+++ Load pickle file named n +++")
        with open('pickle/n.pkl','rb') as f:
          n = pickle.load(f)
    except Exception as e:
        print("Error: %r" % (e))
        exit()
    
     
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
    
if __name__ == "__main__":
    main()
