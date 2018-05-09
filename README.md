Character recognition - (0-9)

There are 4 subfolders in this folder.

Scripts -- contains python scripts for implementing neural network with 1 hidden layer and 1 node in output layer.
           This neural netork can classify two class problem with good accuracy.
        --There is one subfolder called characters -- This contains all the preprocessed character images from 0-9 named
                                              pp0-pp9 respectively.
        --ann.py - this is a library which has necessary functions to train the neural network
        --train.py -This script is used to train neural network for given digit. It extracts images from preprocessed 
                    image set and goes through a number of iterations as specified in the code, modifies weights and
                    stores it in ../w/ folder. It also shows a graph displaying change in cost function over iterations.
                    usage -- python train.py '(0-9)' 
                  
videoProcessing --  Contains script for video recognition. 
                    Continuously preprocess the video frame and feed it to neural network and out the probability
                    of digit specified in argv[1]
                    usage -- python video.py '(0-9)'
                    
w -- contains weight matrices of trained neural network. w1_(digitNumber), w2_(digitNumber)
b -- contains biases for hidden nodes(b1_(digitNumber)) and output nodes(b2_(digitNumber))
                  
