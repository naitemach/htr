from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import os
from WordSegmentation import wordSegmentation, prepareImg
import shutil



path = '../SegWords'



def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	return (recognized[0], probability)

def main():
	Infilename = sys.argv[1]
	img = prepareImg(cv2.imread(Infilename), 50)
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	os.mkdir(path)
	
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		cv2.imwrite('../SegWords/%d.png'%j, wordImg) 

	files = []
	for filename in sorted(os.listdir(path)):
		files.append(os.path.join(path,filename))
	
	decoderType = DecoderType.WordBeamSearch
	model = Model(open('../model/charList.txt').read(), decoderType, mustRestore=True)
	predicted=[]
	probability=[]
	
	for fp in files:
		pred, prob = infer(model,fp)
		predicted.append(pred)
		probability.append(prob)
	
	shutil.rmtree(path)
	print('The predicted sentence is : ',end="'")
	for pw in predicted:
		print(pw, end=" ")
	print("'")

	print('The average probability is : ',end="")
	sum = 0
	for prob in probability:
		sum += prob
	print(sum/len(files)*100)


if __name__ == '__main__':
	main()