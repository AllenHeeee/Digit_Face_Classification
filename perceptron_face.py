from __future__ import division
import numpy
import math
import const


global yitcpt
weightls = []
score = []

def get_feature(data):
    ans = []
    for i in range(const.FaceHeight):
        for j in range(const.FaceWidth):
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
    
    for i in range(2):
        weightls.append(numpy.zeros(4200))

    list = [weightls[0], weightls[1]]
    yitcpt = [0, 0]
    
    for i in range(trainingSize):
        feature = get_feature(combined_data[i])
        global score
        score = []
        for j in range(2):
            score.append( numpy.dot(list[j], feature) + yitcpt[j])
        if not(combined_label[i] == score.index(max(score))):
            list[score.index(max(score))] = numpy.subtract(list[score.index(max(score))],feature.transpose())
            list[combined_label[i]] = numpy.add(list[combined_label[i]],feature.transpose())
            yitcpt[score.index(max(score))] -= 1
            yitcpt[combined_label[i]] += 1


def training(trainingSize):
    global weight
    randomnum = numpy.random.choice(trainingSize, trainingSize, replace=False)
    combined_data = randomize(randomnum, const.TrainingImages)
    combined_label = randomize(randomnum, const.TrainingLabels)
    get_weight(combined_data, combined_label, trainingSize)


def get_face_class(feature):
    local_answer = []
    for i in range(2):
        local_answer.append(0)
    for i in range(2):
        local_answer[i] = numpy.dot(list[i], feature) + yitcpt[i]
    answer = {'0':local_answer[0],'1':local_answer[1]}
    predicted = max(answer, key=answer.get)
    return int(predicted)


def get_predicted_face():
    predicted=[]
    for i in range(len(const.TestImage)):
         predicted.append(get_face_class(get_feature(const.TestImage[i])))
    return predicted

