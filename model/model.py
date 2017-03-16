import tensorflow as tf 
import numpy as np
from tensorflow.contrib import rnn
os.chdir('..')
from data import Data 



LEARNING_RATE = 1e-3
EPOCH = 10000
NUM_CLASSES = 5
NUM_HIDDEN = 128 # commonly chosen hyperparam
DROPOUT_RATE = 0.5

data = Data("imdb_data", "dictionary.txt", MODEL_NAME)

sentence = tf.placeholder("float", [None, EMBEDDING_SIZE, data.maxLength])
label = tf.placeholder("int32", [None, NUM_CLASSES])

weights = {
	"weight": tf.Variable(tf.random_normal([2 * NUM_HIDDEN, NUM_CLASSES]))
}

biases = {
	"biases": tf.Variable(tf.random_normal([NUM_CLASSES]))
}

