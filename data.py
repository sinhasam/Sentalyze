import os
import re
from preprocessing import Processing


class Data(object):
	def __init__(self, relPath, dataFileName, modelFileName, labelFileName):
		self.dataFileName = dataFileName
		self.modelFileName = modelFileName
		self.relPath = relPath
		self.labelFileName = labelFileName
		self.dataList = []
		self.indexList = []
		self.labelList = []
		self.prepareData()
		self.prepareLabels()
		self.currentSentenceIndex = 0
		self.processing = Processing(relPath, self.dataFileName, self.modelFileName, self.maxLength)


	def __str__(self):
		return str(self.maxLength)


	def prepareLabels(self):
		with open(self.labelFileName, 'r') as file:
			for line in file: 
				match = re.match("([0-9]*)\|([0-9]*$)", line)
				rating = float(match.group(2))

				label = lambda rating: int(rating * 5) # 0 for worst - 4 for best

				self.labelList.append(label(rating))


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
		
