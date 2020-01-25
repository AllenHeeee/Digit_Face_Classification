from __future__ import division
import const
import numpy
import math


prior_prob = []
prob = []
count = []
count2 = []


'''
This method is used to shuffle the training label and training image
'''
def randomize(randomList, data):
    ans = []
    for i in range(len(randomList)):
        ans.append(data[randomList[i]])
    return ans

'''
This method is used to calculate the prior probability of digits
'''
def get_prior(data, trainingSize):
    global prior_prob
    global count
    
    for i in range(10):
        count.append(0)
        prior_prob.append(numpy.float64(0))

    for i in range(trainingSize):
        count[data[i]] += 1
        
    for i in range(10):
        prior_prob[i] = numpy.float64(count[i] / trainingSize)

'''
This method is used to extract features from images
'''
def get_feature(data):
    ans = []
    for i in range(const.DigitHeight):
        for j in range(const.DigitWidth):
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
    global count2
    for i in range(10):
        count2.append(numpy.zeros(const.DigitHeight * const.DigitWidth))
        prob.append(numpy.zeros(const.DigitHeight * const.DigitWidth))

    for i in range(trainingSize):
        feature = get_feature(data[i])
        for j in range(len(feature)):
            if feature[j] == 1:
                count2[label[i]][j] += 1

    for i in range(const.DigitHeight * const.DigitWidth):
        for j in range(10):
            prob[j][i] = numpy.float64((count2[j][i] + 0.01) / (count[j] + 0.01))

'''
The main method is called to train from training data
'''
def training(trainingSize):
    global prior_prob
    randomnum = numpy.random.choice(trainingSize, trainingSize, replace=False)
    combined_data = randomize(randomnum, const.TrainingImages)
    combined_label = randomize(randomnum, const.TrainingLabels)
    get_prior(combined_label, trainingSize)
    get_prob(combined_data, combined_label, trainingSize)


'''
This method is used to predict the the certain digit
'''
def get_digit_class(features):
    global prob
    global prior_prob
    miss = 0.0000000000001
    local_prob = []
    tempCount = []
    for i in range(10):
        tempCount.append(0)
        local_prob.append(0)
    for i in range(len(features)):
        if features[i] == 1:
            for j in range(10):
                tempCount[j] += math.log(prob[j][i])
        else:
            for j in range(10):
                if(prob[j][i] == 1):
                    prob[j][i] -= miss
            for j in range(10):
                tempCount[j] += math.log(1-prob[j][i])
                
    for i in range(10):
        local_prob[i] = math.log(prior_prob[i])+tempCount[i]

    digit_class = {
        '0':local_prob[0],
        '1':local_prob[1],
        '2':local_prob[2],
        '3':local_prob[3],
        '4':local_prob[4],
        '5':local_prob[5],
        '6':local_prob[6],
        '7':local_prob[7],
        '8':local_prob[8],
        '9':local_prob[9],
    }
    return max(digit_class, key = digit_class.get)
    
'''
This method calculates and returns the predicted results
'''
def get_predicted_digit():
    digit = []
    for i in range(len(const.TestImage)):
        digit.append(get_digit_class(get_feature(const.TestImage[i])))
    return digit

