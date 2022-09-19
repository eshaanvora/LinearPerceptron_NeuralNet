#Eshaan Vora
#EshaanVora@gmail.com

#Neural Networks: Linear Perceptron Implementation
#This program will run AS MANY EPOCHS AS NEEDED until the weights do not update anymore, indicating that training is complete

#Goal is to learn the weights, such that every feature, x, is mapped to the correct label
#Use update rule to update weights, until labels are mapped correctly


#Weights are initialized to 0.4
w0 = -.4
w1 = .4
w2 = -.4
weights = [w0, w1, w2]

#Initialize inputs matrix for x0,x1,x2 and truth labels
inputs = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]
labels = [1,1,1,-1]

#Function to calculate weights after a given row of inputs (given X = row of inputs)
#Updating weight after 4th row represents 1 epoch (because we have iterated through each input)
def getWeights(row, weight, input, label):
    #Initialize learning rate
    learning_rate = 0.01
    #Initialize variables used for calculations
    dot_product = 0
    output = 0

    #Access updated weight variables
    w0 = weight[0]
    w1 = weight[1]
    w2 = weight[2]

    #Calculate Dot Product and encode output
    for index, i in enumerate(input[row]):
        dot_product = dot_product + (i * weight[index])
    if dot_product > 0:
        output = 1
    else:
        output = -1

    #Adjust weights using the update rule
    #As seen in the Update Rule, if truth and output are the same, the change in weights would be 0
    #As seen in the Update Rule, if input is 0, the change in weights would be 0
    delta_w0 = learning_rate * (label[row] - output) * input[row][0]
    w0 = w0 + delta_w0

    delta_w1 = learning_rate * (label[row] - output) * inputs[row][1]
    w1 = w1 + delta_w1

    delta_w2 = learning_rate * (label[row] - output) * inputs[row][2]
    w2 = w2 + delta_w2

    #Update weights
    #Round weights before updating for legibility
    weights = [round(w0,6), round(w1,6), round(w2,6)]
    #weights = [w0,w1,w2]

    #Print updated weights after each input in each Epoch
    print("updated weight at X=",inputs[row],":",weights,)

    #Return updated weights
    return weights

#Start first epoch by generating weights after first row of data
wghts = getWeights(0, weights, inputs, labels)

#Complete first epoch by generating weights after first row until end of first epoch
for i in range(1,4):
    temp_wghts = wghts
    wghts = getWeights(i, temp_wghts, inputs, labels)

print("VALUE OF WEIGHTS AT EPOCH 1 : ", wghts)

#While loop to keep calculating/updating weights until weights do not change anymore (loop until convergence)
epoch_counter = 2
condition = True
while(condition):
    previous_epoch_weight = wghts
    wghts = getWeights(0, wghts, inputs, labels)

    #Complete epoch by updating weights
    for j in range(1,4):
        temp_wghts = wghts
        wghts = getWeights(j, temp_wghts, inputs, labels)

    #Check if we need to update weights again; if weights do not change since last epoch, we do not need additional updates (reached convergence)
    if previous_epoch_weight[0] == wghts[0] and previous_epoch_weight[1] == wghts[1] and previous_epoch_weight[2] == wghts[2]:
        condition = False
        break

    print("VALUE OF WEIGHTS AT EPOCH", epoch_counter, ":" , wghts)

    epoch_counter += 1

print("\nTOTAL NUMBER OF EPOCHS:",epoch_counter-1,"\n")
