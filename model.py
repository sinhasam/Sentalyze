import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim



NUM_HIDDEN = 300 
DROPOUT_RATE = 0.5
EMBEDDING_SIZE = 5
BATCH_SIZE = 1
NUM_FILTERS = 1
NUM_CLASSES = 5
NUM_LAYERS = 3
OUT_CHANNELS = 100


class Net(nn.Module):
    def __init__(self, maxSentenceLength, inputSize, weights=None):
        
        super(Net, self).__init__()
        self.inputSize = inputSize
        self.maxSentenceLength = maxSentenceLength
        self.weights = None

        self.train = True
        self.setLayers()


    def setLayers(self):
        self.LSTM = nn.LSTM(input_size=self.inputSize, hidden_size=NUM_HIDDEN, bidirectional=True, 
                        num_layers=NUM_LAYERS, dropout=DROPOUT_RATE, bias=True)
        
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=OUT_CHANNELS, kernel_size=3, bias=True), 
                                        nn.BatchNorm2d(OUT_CHANNELS),
                                        nn.ReLU())

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # 299 because embedding dim is with a kernel_stride of 3 results in (300 - 2) = 298 
        # and 27 because max sentence length is 56 and after the convlayer and maxpool it is 27
        self.fcLayer = nn.Linear(27*299*OUT_CHANNELS*BATCH_SIZE, NUM_CLASSES) 

        self.softmax = nn.LogSoftmax()



    def forward(self, in1):

        # (sequence_length, batch_size, input_size)
        # (num_layers*2, batch, hidden_size)
        hiddenVariable = Variable(torch.randn(NUM_LAYERS*2, BATCH_SIZE, NUM_HIDDEN), requires_grad=True) # "*2" is for num_directions which = 2 since it is bidirectional
        # (num_layers*2, batch, hidden_size)
        cellVariable = Variable(torch.randn(NUM_LAYERS*2, BATCH_SIZE, NUM_HIDDEN), requires_grad=True)


        #states is a tuple with (hidden_states, cell_states)
        output, _ = self.LSTM(in1, (hiddenVariable, cellVariable))
        
        dimSize = output.size()
        
        output = output.resize(BATCH_SIZE, NUM_FILTERS, dimSize[0], dimSize[2]) # to make it 4d to perform 2 d convolutions

        output = self.convLayer(output)

        output = self.maxpool(output)


        output = output.view(-1, 27*299*OUT_CHANNELS*BATCH_SIZE)

        output = self.fcLayer(output)

        return self.softmax(output)

    







