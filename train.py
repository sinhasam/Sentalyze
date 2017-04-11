import torch
import os
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from model import Net
import torch.nn as nn
import sys

from preprocessing import Processing

from data import Data

NUM_EPOCH = 100
LEARNING_RATE = 1e-3
NUM_CLASSES = 5
EMBEDDING_SIZE = 300

data = Data("imdb_data", "dictionary.txt", "NEED_TO_DOWNLOAD", "sentiment_labels.txt")
numSentence = data.numSentences

model = Net(data.maxLength, EMBEDDING_SIZE)

lossFunction = nn.NLLLoss() # dont need cross entropy since we do softmax in the model

optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

for epoch in range(NUM_EPOCH):
	print("On epoch " + epoch)

	model.zero_grad()
	sentence, sentenceLen, label = getNextSentence()
	sentence = torch.from_numpy(sentence)
	
	logProb = model.forward(sentence)

	loss = lossFunction(logProb, label)

	loss.backward()
	optimizer.step()

