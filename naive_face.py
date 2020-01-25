from __future__ import division
import const
import numpy
import math

none_face_count = 0
face_count = 0

# numbers on index 0 is the prior prob for not face
# numbers on index 1 is the prior prob for face
prior_prob = []

# numbers on index 0 is the prob for not face
# numbers on index 1 is the prob for face
prob = []
pixel_count = []

'''
This method is used to count how many pictures are face in the training label
'''
def count_face(labels, trainingSize):
	global none_face_count
	global face_count
	for i in range(0,trainingSize):
		if labels[i] == 0:
			none_face_count += 1
		else:
			face_count += 1
		
'''
This method is used to shuffle the training label and training image
'''
def randomize(randomList, data):
	ans = []
	for i in range(len(randomList)):
		ans.append(data[randomList[i]])
	return ans

'''
This method is used to calculate the prior probability of not face and contained face
'''
def get_prior(trainingSize):
	global prior_prob
	global none_face_count
	global face_count
	
	prior_prob.append(numpy.float64(none_face_count/trainingSize))
	prior_prob.append(numpy.float64(face_count/trainingSize))
	

'''
This method is used to extract features from images
'''
def get_feature(data):
	ans = []
	for i in range(const.FaceHeight):
		for j in range(const.FaceWidth):
			if data[i][j] != 0:
				ans.append(1)
				continue
			ans.append(0)
	return numpy.asarray(ans)
		

'''
This method is used to calculate the probability
'''
def get_prob(data, label, trainingSize):
	global prob
	global pixel_count
	global none_face_count
	global face_count
	
	count = []
	count.append(none_face_count)
	count.append(face_count)
	
	for i in range(2):
		pixel_count.append(numpy.zeros(const.FaceHeight * const.FaceWidth))
		prob.append(numpy.zeros(const.FaceHeight * const.FaceWidth))

	for i in range(trainingSize):
		feature = get_feature(data[i])
		for j in range(len(feature)):
			if feature[j] == 1:
				pixel_count[label[i]][j] += 1

	for i in range(const.FaceHeight * const.FaceWidth):
		for j in range(2):
			prob[j][i] = numpy.float64((pixel_count[j][i] + 0.01) / (count[j] + 0.01))

'''
This method calculates and returns the predicted results
'''
def get_predicted_face():
	digit = []
	for i in range(len(const.TestImage)):
		digit.append(get_digit_class(get_feature(const.TestImage[i])))
	return digit

'''
This method is used to predict the whether the certain image contains face
'''
def get_digit_class(features):
	global prob
	global prior_prob
	miss = 0.0000000000001
	local_prob = []
	tempCount = []
	for i in range(2):
		tempCount.append(0)
		local_prob.append(0)
	for i in range(len(features)):
		if features[i] == 1:
			for j in range(2):
				tempCount[j] += math.log(prob[j][i])
		else:
			for j in range(2):
				if(prob[j][i] == 1):
					prob[j][i] -= miss
			for j in range(2):
				tempCount[j] += math.log(1-prob[j][i])
				
	for i in range(2):
		local_prob[i] = math.log(prior_prob[i])+tempCount[i]

	face_class = {
		'0':local_prob[0],
		'1':local_prob[1]
	}
	return max(face_class, key = face_class.get)

'''
The main method is called to train from training data
'''
def training(trainingSize):
	global none_face_count
	global face_count
	randomnum = numpy.random.choice(trainingSize, trainingSize, replace=False)
	combined_data = randomize(randomnum, const.TrainingImages)
	combined_label = randomize(randomnum, const.TrainingLabels)
	count_face(combined_label,trainingSize)
	get_prior(trainingSize)
	get_prob(combined_data, combined_label, trainingSize)
	