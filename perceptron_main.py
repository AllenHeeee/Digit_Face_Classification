import time
import const
import perceptron_digit
import perceptron_face
import numpy


prior_probb = []
if __name__ == "__main__":
    # Main Function of classifier
    print("Start Scanning Digits\n")
    results = []
    for i in range(0, 10):
        results = []
        timelist = []
        for j in range(5):
            print("Digit Iteration: "+str(5*i+j+1))
            traingingSize = 500 * (i+1)
            const.load_digit(traingingSize)
            
            print("Training...")
            print("Training Size: "+str(traingingSize))
            start_time = time.time()
            # start training
            perceptron_digit.training(traingingSize)
            end_time = time.time()
            print("Training Complete")
            print("Training Time: "+str(end_time-start_time)+" Sec")
            print("Testing...")
            predicted_digit = perceptron_digit.get_predicted_digit()
            timelist.append(end_time-start_time)
            results.append(const.calculate_prediction(predicted_digit, const.TestLabels))
            print ("Correct Prediction: " + str(results[j])+"%")
            print("\n")

        print("Digit Summary:")
        print("Training Size = "+str((i+1)*10)+"%")
        print("Mean of Training Time = "+str(numpy.mean(timelist))+" Sec")
        print("Mean of Correct Prediction = "+str(numpy.mean(results))+"%")
        print("Std of Correct Prediction = "+str(numpy.std(results))+"%")
        print("\n\n")
    
    
    
    print("Start Scanning Faces\n")
    for i in range(0, 10):
        results = []
        timelist = []
        for j in range(5):
            print("Face Iteration: "+str(5*i+j+1))
            trainingSize = 45 * (i+1)
            const.load_face(trainingSize)
            print("Training...")
            print("Training Size: "+str(trainingSize))
            start_time = time.time()
            # start training
            perceptron_face.training(trainingSize)
            end_time = time.time()
            print("Training Complete")
            print("Training Time: "+str(end_time-start_time)+" Sec")
            predicted_face = perceptron_face.get_predicted_face()
            timelist.append(end_time-start_time)
            results.append(const.calculate_prediction(predicted_face, const.TestLabels))
            print ("Correct Prediction: " + str(results[j])+"%")
            print("\n")
        print("Face Summary:")
        print("Training Size = "+str((i+1)*10)+"%")
        print("Mean of Training Time = "+str(numpy.mean(timelist))+" Sec")
        print("Mean of Correct Prediction = "+str(numpy.mean(results))+"%")
        print("Std of Correct Prediction = "+str(numpy.std(results))+"%")
        print("\n\n")
    

        
        
        
