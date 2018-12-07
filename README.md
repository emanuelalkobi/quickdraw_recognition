# Quickdraw recognition

### Interactive web site:

http://wym613.github.io

### What is Doodle recognition?

Doodle recognition project is performed by a classifier that takes the user input, given as a sequence of strokes of points in x and y, and recognizes the object category that the user tried to draw. 

### Our task in the project?

Our main task is to receive a drawing as  input. And predict the highest success rate what is drawn in it from a list of labels which we trained our algorithm over it. The labels are consist of 345 categories from different subjects like animals, food or furniture. Then designing a website to show the result.

### Data set ?


The Quick Draw Dataset is a collection of 50 million drawings across 345 categories. This draws were created by players of the game Quick Draw!

Each draw is a 28 by 28  matrix grayscale bitmap in numpy .npy format.


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



<img src="/process_img/apple.jpg" width="100" height="100" style="width:80%"> <img src="/process_img/axe.jpg.jpg" width="100" height="100" style="width:80%"> <img src="/process_img/cat.jpg" width="100" height="100" style="width:80%">


	



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
     - 91 stuff categories




The function that we are minimizing using the Adam algorithm  is the sparse softmax cross entropy between logits(output of the CNN) and the truth labels.

For the final result we apply a softmax function to get probabilities  for each class and we predicted the class as the class that has the highest probability.





#data
data is downloaded as numpy arrays from:
https://github.com/googlecreativelab/quickdraw-dataset
