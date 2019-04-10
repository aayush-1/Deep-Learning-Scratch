README:

Command to run:
 python cnn_scratch.py

command to run with provided input file("inp.txt")
 python cnn_scratch.py <inp.txt	




help sites
https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
https://codereview.stackexchange.com/questions/133251/a-cnn-in-python-without-frameworks
https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c




Input file is shared as inp.txt

reference input format:

input image path
enter no. of convolutional layers
enter non linearity functions(tanh or sigmoid) for corresponding layer
enter stride for corresponding layer
enter padding (same or valid) for corresponding layer
filter kernel height
filter kernel width
enter no. of filters for convolution for each layer
pooling kernel height
pooling kernel width
enter pool type(max or avg) for each conv layer
enter pool stride for each conv layer
enter input size for MLP
enter hidden layers for MLP
size of corresponding hidden layer
non linearity fn(tanh or sigmoid) for corresponding layer
enter output_size for MLP

