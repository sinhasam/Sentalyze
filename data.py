import os
import re
from preprocessing import Processing


class Data(object):
	def __init__(self, relPath, dataFileName, word2VecFileName, labelFileName):
		self.dataFileName = dataFileName
		self.word2VecFileName = word2VecFileName
		self.relPath = relPath
		self.labelFileName = labelFileName
		self.dataList = []
		self.indexList = []
		self.labelList = []
		self.prepareData()
		self.prepareLabels()
		self.currentSentenceIndex = 0
		self.processing = Processing(self.word2VecFileName, self.maxLength)


	def __str__(self):
		return str(self.maxLength)


	def prepareLabels(self):
		with open(self.labelFileName, 'r') as file:
			for count, line in enumerate(file): 
				if count == 0: # the first line is not a label
					continue
				match = re.match("([0-9]*)\|(0\.[0-9]*)", line)
				if not match:
					rating = 0.9
				else:
					rating = float(match.group(2))

				self.labelList.append(rating)

			self.labelList = list(map(lambda rating: int(rating * 5), self.labelList)) # 0 for worst - 4 for best


	def prepareData(self):
		self.maxLength = 0
		os.chdir(self.relPath)
		
		with open(self.dataFileName, 'r') as file:
			
			for count, line in enumerate(file):
				
				match = re.match("(.*)\|([0-9]*$)", line)
				line = match.group(1)
				index = int(match.group(2))
				line = line.split()

				self.maxLength = max(self.maxLength, len(line))
				self.dataList.append(line)
				self.indexList.append(index)
			
			self.numSentences = count	
			

	
	def getNextSentence(self):
		nextLine = self.dataList[self.currentSentenceIndex]
		vecSentence, sentenceLen = self.processing.sentence2vec(nextLine)
		label = self.getNextLabel()
		self.currentSentenceIndex += 1
		return vecSentence, label, sentenceLen


	def getNextLabel(self): 
		index = self.indexList[self.currentSentenceIndex]
		return self.labelList[index]
		
