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
NUM_HIDDEN = 128 # commonly chosen hyperparam
DROPOUT_RATE = 0.5
EMBEDDING_SIZE = 5



class Net(nn.Module):
    def __init__(self, maxSentenceLength, embedDim, inputSize, weights = None):
        
        super(Net, self).__init__()
        
        self.inputSize = inputSize
        self.embedDim = embedDim
        self.maxSentenceLength = maxSentenceLength
        
        self.input = [Variable(torch.tensor(1, self.embedDim)) for _ in range(self.maxSentenceLength)]
        self.hiddenParam = Variable(torch.randn(1, 1, self.NUM_HIDDEN)) #fix the dimensions

        if not weights:
           self.setBLSTM()

        else: 
            weights = weights


    def setBLSTM(self):
        self.lstm = nn.LSTM(input_size=self.inputSize,hidden_size=NUM_HIDDEN,bidirectional=True,
                        dropout=DROPOUT_RATE, bias=True)
        


    def forward(self):
        pass