from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np


EMBEDDING_SIZE = 300 

class Processing(object):
	def __init__(self, modelFilename, maxLength):
		print("Loading " + modelFilename)
		self.word2Vec = KeyedVectors.load_word2vec_format(modelFilename, binary=True)
		print("Done loading")
		self.maxLength = maxLength


	def sentence2vec(self, sentence):
		word2sentenceMatrix = np.zeros((self.maxLength, EMBEDDING_SIZE))
		print(sentence)
		try:
			word2sentenceMatrix = [self.word2Vec[word] for word in word_tokenize(str(sentence))]

		except:
			return None, 0

		word2sentenceVec = word2sentenceMatrix.reshape(1,-1)


		return word2sentenceVec, len(word_tokenize(sentence))