from __future__ import division
import numpy
import math
import const

global yitcpt
weightls = []
score = []

def get_feature(data):
    ans = []
    for i in range(const.DigitHeight):
        for j in range(const.DigitWidth):
            if data[i][j] != 0:
                ans.append(1)
                continue
            ans.append(0)
    return numpy.asarray(ans)

def randomize(randomList, data):
    ans = []
    for i in range(len(randomList)):
        ans.append(data[randomList[i]])
    return ans

def get_weight(combined_data, combined_label, trainingSize):
    global weightls
    global list
    global yitcpt
    for i in range(10):
        weightls.append(numpy.zeros(784))
    
    list = [weightls[0], weightls[1],weightls[2],weightls[3],weightls[4],weightls[5],
        weightls[6], weightls[7], weightls[8], weightls[9]]
    yitcpt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    for i in range(trainingSize):
        feature = get_feature(combined_data[i])
        global score
        score = []
        for j in range(10):
            score.append( numpy.dot(list[j], feature) + yitcpt[j])
        
        if not(combined_label[i] == score.index(max(score))):
            coeff = (numpy.dot((list[score.index(max(score))] - list[combined_label[i]]),feature) + 1) / (2*numpy.dot(feature,feature))
            coeff = numpy.absolute(coeff)
            c = 0.001
            coeff = min(c,coeff)
            #print(coeff)
            list[score.index(max(score))] = numpy.subtract(list[score.index(max(score))],coeff*feature.transpose())
            list[combined_label[i]] = numpy.add(list[combined_label[i]],coeff*feature.transpose())
            yitcpt[score.index(max(score))] -= coeff
            yitcpt[combined_label[i]] += coeff




def training(trainingSize):
    global weight
    randomnum = numpy.random.choice(trainingSize, trainingSize, replace=False)
    combined_data = randomize(randomnum, const.TrainingImages)
    combined_label = randomize(randomnum, const.TrainingLabels)
    get_weight(combined_data, combined_label, trainingSize)


def get_digit_class(feature):
    local_answer = []
    for i in range(10):
        local_answer.append(0)
    for i in range(10):
        local_answer[i] = numpy.dot(list[i], feature) + yitcpt[i]
        
    answer = {'0':local_answer[0],'1':local_answer[1],'2':local_answer[2],'3':local_answer[3],'4':local_answer[4],'5':local_answer[5],'6':local_answer[6],'7':local_answer[7],'8':local_answer[8],'9':local_answer[9]}
    predicted = max(answer, key=answer.get)
    return int(predicted)


def get_predicted_digit():
    predicted=[]
    for i in range(len(const.TestImage)):
         predicted.append(get_digit_class(get_feature(const.TestImage[i])))
    return predicted

