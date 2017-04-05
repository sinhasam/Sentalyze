import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import optim

os.chdir('..')
from data import Data 



LEARNING_RATE = 1e-3
EPOCH = 10000
NUM_CLASSES = 5
NUM_HIDDEN = 300 
DROPOUT_RATE = 0.5
EMBEDDING_SIZE = 5
BATCH_SIZE = 1


class Net(nn.Module):
    def __init__(self, maxSentenceLength, embedDim, inputSize, weights = None):
        
        super(Net, self).__init__()
        self.inputSize = inputSize
        self.embedDim = embedDim
        self.maxSentenceLength = maxSentenceLength
        
        if not weights:
           self.weights = None
           self.train = True
           self.setLayers()
        else: 
            self.weights = weights


    def setLayers(self):
        self.LSTM = nn.LSTM(input_size=self.inputSize,hidden_size=NUM_HIDDEN,bidirectional=True, 
                        num_layers=1, dropout=DROPOUT_RATE, bias=True)
        
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, bias=True), 
                                        nn.BatchNorm2d(100),
                                        nn.ReLU())
        # nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, bias=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fcLayer = nn.Linear(4455100, NUM_CLASSES) # what the numbers comes out to
        self.softmax = nn.LogSoftmax()


    def forward(self, in1):
        # (sequence_length, batch_size, input_size)
        # self.in1 = Variable(torch.randn(self.embedDim, BATCH_SIZE, self.inputSize)) 
        # (num_layers*2, batch, hidden_size)
        self.hiddenVariable = Variable(torch.randn(2, BATCH_SIZE, NUM_HIDDEN))
        # (num_layers*2, batch, hidden_size)
        self.cellVariable = Variable(torch.randn(2, BATCH_SIZE, NUM_HIDDEN))

        #states is a tuple with (hidden_states, cell_states)
        self.output, self.states = self.LSTM(self.in1, (self.hiddenVariable, self.cellVariable))
        
        dimSize = self.output.size()
        
        self.output = self.output.resize(BATCH_SIZE, dimSize[1], dimSize[0], dimSize[2]) # to make it 4d to perform 2 d convolutions

        self.output = self.convLayer(self.output)

        self.output = self.maxpool(self.output)
        
        self.output = self.output.view(self.output.size(0), -1)

        self.output = self.softmax(self.output)


    







