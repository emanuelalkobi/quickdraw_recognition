# Quickdraw recognition

## How to run our code?

### CNN 
In the CNN directory there  are all the files regard the CNN.

train.py:
 
   *   This file train the CNN with the next parameters:
       - epoch: number of epochs to run
        
       - draws per class: how much draws to use for training per class ( in % ,a number between 0 to 100)
         
       - class number:how much classes to train over, it will take the N classes alphabetically 
    
The command will be :
python train.py --epoch <epoch number> -draws_per_class <percentage per class> -class_num <class number to train>
    
   *   In a case that no arguments will be set the next default arguments will run :
       - epoch:10
       - draws per class:30
       - class number:30
       
test_our.py:
       
   *   This file test 1 draw  with the last CNN  model which was train.
    
The command will be :
    python test_our.py --draw_path <draw path to identify>  --label <label of the current draw-truth label between 1 to 30>

test.py:
This file is an helper file and his main target his to run the validation draws over the CNN during training.It is also used the run the test_our.py file.

The CNN/saved_see directory is used to save and store the trained model after training.
The result of 100 class is in /CNN/CNN_100class.ipynb. The accuracy of testing 100 classes is 0.93.

### LSTM 

In the LSTM folder there are two files. One is for training 30 classes and the other is for training 100 classes. By using and testing the model, you need to install keras on laptop. 

When you try to testing small number of classes, run the jupyter notebook in LSTM/LSTM_30class.ipynb. If you want to try more dataset, you could run LSTM.LSTM_100class.ipynb. You could add more dataset by downloading more  from the google cloud.

To train and test LSTM model for 100 classes, run /LSTM/LSTM_100class.ipynb. The accuracy of testing 100 classes is 0.62.

### Interactive web site:

http://wym613.github.io


We build a website which embedded our model using TensorFlow.js. 
The code fot the website is stored here: https://github.com/wym613/wym613.github.io

To implement the model to the website  we used Keras to generate the model.h5. And transform it to .json file by Tensorflow.js. 
Fianlly, to combine the backend and frontend, we use github.io to realize the website. You can see our demo vedio here:

https://www.youtube.com/watch?v=4kolOGHQUC8&feature=youtu.be

### What is Doodle recognition?

Doodle recognition project is performed by a classifier that takes the user input, given as a sequence of strokes of points in x and y, and recognizes the object category that the user tried to draw. 

### Our task in the project?

Our main task is to receive a drawing as  input. And predict the highest success rate what is drawn in it from a list of labels which we trained our algorithm over it. The labels are consist of 345 categories from different subjects like animals, food or furniture. Then designing a website to show the result.

### Data set ?


The Quick Draw Dataset is a collection of 50 million drawings across 345 categories. This draws were created by players of the game Quick Draw!

Each draw is a 28 by 28  matrix grayscale bitmap in numpy .npy format.

All the data set was downloaded from :

https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap


*   For our project we picked 30 different categories:
    - Ambulance
    - Apple
    - Basketball
    - Bicycle
    - Boomearang
    - Butterfly
    - Car
    - Carrot
    - Cat
    - Chair
    - Clock
    - Cookie
    - Cup
    - Donut
    - Envelope
    - Flower
    - Key
    - Knife
    - Lighting
    - Pencil
    - Pizza
    - Rainbow
    - Snake
    - Spider
    - Star
    - Tractor
    - Tree
    - Whale
    - Windmill
    
* All the 30 calsses data set are stored at data directory.

* At test_img directory there are few draws that we create to test our models.

* We also tested 100 classes. The categories can be seen in /LSTM/LSTM_100class.ipynb and /CNN/CNN_100class.ipynb.







### Process data

The data is available in few formats : ndjson,bin and npy. We use the .npy data format.

In this format, the drawings are rendered to a 784 vector size. We reshape  this vector to a 28 by 28 matrix and then flip the value of the matrix (255-value) in order to get better result for our own drawings.


 <img src="/test_img/cat_pre.jpg" width="100" height="100" style="width:80%">  <img src="/test_img/cat.jpg" width="100" height="100" style="width:80%">	
 
 <img src="/test_img/axe_pre.jpg" width="100" height="100" style="width:80%"><img src="/test_img/axe.jpg" width="100" height="100" style="width:80%">
 
  <img src="/test_img/apple_pre.jpg" width="100" height="100" style="width:80%"><img src="/test_img/apple.jpg" width="100" height="100" style="width:80%">



### Approach of the project?

CNN - Convolutional neural network

LSTM - Long short-term memory


### CNN
We build a convolutional neural network in order to train our quick draw classifier.

*    After few tries we found that the best Cnn setup is the following network:
     - Convolutional layer with 32 filters,kernel size of 5 by 5  and padding 
     - Max pool layer with size 2 by 2 and a stride of 2
     - Flatten layer 
     - Hidden layer with 500 units
     - RelU activation function
     - Flatten layer to the number of different classes.


The function that we are minimizing using the Adam algorithm  is the sparse softmax cross entropy between logits(output of the CNN) and the truth labels.

For the final result we apply a softmax function to get probabilities  for each class and we predicted the class as the class that has the highest probability.


### LSTM
LSTMs don’t have a fundamentally different architecture from RNNs, but they use a different function to compute the hidden state. The memory in LSTMs are called cells and you can think of them as black boxes that take as input the previous state  and current input. 

Internally, these cells decide what to keep in (and what to erase from) memory. They then combine the previous state, the current memory, and the input. It turns out that these types of units are very efficient at capturing long-term dependencies.









