
'''
Some global vatiables 
'''
TrainingImages = []

TrainingLabels = []

ValidationImages = []

ValidationLabels = []

TestImage = []

TestLabels = []

DigitWidth = 28

DigitHeight = 28

FaceWidth = 60

FaceHeight = 70


'''
Convert pixel to digit or convert digit to pixel
'''
def convert_pixel_between_int(pixel):
    if isinstance(pixel, int):
        switcher = {
            0: ' ',
            1: '#',
            2: '+'
        }
        return switcher.get(pixel, 0)
    else:
        switcher = {
            ' ': 0,
            '#': 1,
            '+': 2
        }
        return switcher.get(pixel, 0)

'''
This method is going to read the files and load data 
'''
def load_file(filename, num, is_label_file, height):
    data = [l[:-1] for l in open(filename).readlines()]
    ans = []
    if not is_label_file:
        data.reverse()
    for i in range(num):
        if is_label_file:
            ans.append(int(data[i]))
        else:
            temp = []
            for j in range(height):
                temp.append(map(convert_pixel_between_int, list(data.pop())))
            ans.append(temp)
    return ans

'''
This method is used to load digit files
'''
def load_digit(trainingSize):
    global TrainingImages

    global TrainingLabels

    global ValidationImages

    global ValidationLabels

    global TestImage

    global TestLabels

    TrainingImages = load_file("data/digitdata/trainingimages", trainingSize, False, DigitHeight)
    TrainingLabels = load_file("data/digitdata/traininglabels", trainingSize, True, DigitHeight)


    ValidationImages = load_file("data/digitdata/validationimages", 1000, False, DigitHeight)
    ValidationLabels = load_file("data/digitdata/validationlabels", 1000, True, DigitHeight)

    TestImage = load_file("data/digitdata/testimages", 1000, False, DigitHeight)
    TestLabels = load_file("data/digitdata/testlabels", 1000, True, DigitHeight)

'''
This method is used to compare the predicted results with actual results and returns the percentage.
'''
def calculate_prediction(predicted_data, correct_data):
    correct = 0.0
    for j in range(len(predicted_data)):
        num1 = int(correct_data[j])
        num2 = int(predicted_data[j])
        if(num1 == num2):
            correct += 1
    return correct/len(predicted_data)*100


'''
This method is used to load face files
'''
def load_face(trainingSize):
    global TrainingImages

    global TrainingLabels

    global ValidationImages

    global ValidationLabels

    global TestImage

    global TestLabels

    TrainingImages = load_file("data/facedata/facedatatrain", trainingSize, False, FaceHeight)
    TrainingLabels = load_file("data/facedata/facedatatrainlabels", trainingSize, True, FaceHeight)

    ValidationImages = load_file("data/facedata/facedatavalidation", 301, False, FaceHeight)
    ValidationLabels = load_file("data/facedata/facedatavalidationlabels", 301, True, FaceHeight)

    TestImage = load_file("data/facedata/facedatatest", 150, False, FaceHeight)
    TestLabels = load_file("data/facedata/facedatatestlabels", 150, True, FaceHeight)
