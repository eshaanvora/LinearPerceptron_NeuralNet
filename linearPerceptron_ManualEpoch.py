#Eshaan Vora
#EshaanVora@gmail.com

#Neural Networks: Linear Perceptron Implementation
#This program will run a SPECIFIED NUMBER of epochs and print the weights of the linear perceptron at the specified epoch

#Goal is to learn the weights, such that every feature, x, is mapped to the correct label
#Use update rule to update weights, until labels are mapped correctly


# Weights are initialized at 0.2
w0 = 0.2
w1 = 0.2
w2 = 0.2
weights = [w0, w1, w2]
# Create matrix of inputs and labels for AND
inputs = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]
labels = [1,1,1,-1]

# Calculates The Weights At A Given Row
def getWeights(row, weight, input, label):
    # Intitializes Variables - LR, DP, Count, Output
    learning_rate = 0.1
    dot_product = 0
    output = 0

    # Grab Individual Weight Variables
    w0 = weight[0]
    w1 = weight[1]
    w2 = weight[2]

    # Calculate Dot Product And Output
    for index, i in enumerate(input[row]):

        dot_product = dot_product + (i * weight[index])

    if dot_product > 0:
        output = 1
    else:
        output = -1

    # Adjust Weights With Update Formula
    # As seen in Update Formula, if truth and output are the same, the change in weights would be 0
    delta_w0 = learning_rate * (label[row] - output) * input[row][0]
    w0 = w0 + delta_w0

    delta_w1 = learning_rate * (label[row] - output) * inputs[row][1]
    w1 = w1 + delta_w1

    delta_w2 = learning_rate * (label[row] - output) * inputs[row][2]
    w2 = w2 + delta_w2

    # Adds Weights To Array
    weights = [round(w0,1), round(w1,1), round(w2,1)]
    print("updated weight at X=",inputs[row],":",weights,)

    # return the values of adjusted weights
    return weights

def get_Epoch_Weights(numEpochs):

    # Gets weights after first row of training data
    wghts = getWeights(0, weights, inputs, labels)

    # Get weights after first epoch
    for i in range(1,4):
        temp_wghts = wghts
        wghts = getWeights(i, temp_wghts, inputs, labels)

    print("VALUE OF WEIGHTS AT EPOCH 1 : ", wghts)
    # For Loop to Calculate Weights After More Run Throughs

    for k in range(1, numEpochs):

        wghts = getWeights(0, wghts, inputs, labels)

        for j in range(1,4):
            temp_wghts = wghts
            wghts = getWeights(j, temp_wghts, inputs, labels)

        print("VALUE OF WEIGHTS AT EPOCH", k + 1, ":" , wghts)

############## MAIN ###########################################################
get_Epoch_Weights(3)
