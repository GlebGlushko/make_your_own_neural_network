# %mathplotlib inline

import numpy as np
import tkinter

import sys
from PIL import Image
import scipy.special
# import mathplotlib.pyplot
from tkinter import filedialog


class neuralNetwork:

    def __init__(self, input_nodes=0, hidden_nodes=0, output_nodes=0, learning_rate=0.):
        self.inodes = input_nodes
        self.outnodes = output_nodes
        self.hidnodes = hidden_nodes
        self.lrn = learning_rate
        self.weight_in = (np.random.rand(self.hidnodes, self.inodes) - 0.5)
        self.weight_out = (np.random.rand(self.outnodes, self.hidnodes) - 0.5)
        self.activation_function = lambda x: scipy.special.expit(x)

    def set(self, input_nodes, hidden_nodes, output_nodes, learning_rate, weight_in, weight_out):
        self.inodes = input_nodes
        self.outnodes = output_nodes
        self.hidnodes = hidden_nodes
        self.lrn = learning_rate
        self.weight_in = weight_in
        self.weight_out = weight_out

    def get(self):
        return [self.inodes, self.hidnodes, self.outnodes, self.lrn, self.weight_in, self.weight_out]

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        hidden_inputs = np.dot(self.weight_in, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weight_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        targets = np.array(target_list, ndmin=2).T
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weight_out.T, output_errors)
        self.weight_out += self.lrn * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                             np.transpose((hidden_outputs)))
        self.weight_in += self.lrn * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                            np.transpose(inputs))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.weight_in, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        #print(len(self.weight_out))
        # print(len(self.weight_out[0]))
        #print(len(hidden_outputs))
        # print(len(hidden_outputs[0]))

        final_inputs = np.dot(self.weight_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def check():
    print("Do you want to recognize your picture? (Y/N)")
    ch = input()
    if ch == 'N':
        sys.exit(0)
    link = filedialog.askopenfilename()
    image = Image.open(link)
    image = image.resize((28, 28))
    image.save('conv.jpg')
    inputs = np.zeros(784)
    for x in range(28):
        for y in range(28):
            r, b, g = image.getpixel((y, x))
            inputs[28 * x + y] = (1.0 - (r + g + b) / 765.0) * 0.99 + 0.01
    outputs = n.query(inputs)
    for i in outputs:
        print(i, end=' ')
    print()
    print(np.argmax((outputs)))
    check()


'''
def retrain():
    input_nodes = 784
    output_nodes = 10
    hidden_nodes = 100
    learning_rate = 0.3
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    data_file = open("mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()
    epochs = 2
    for e in range(epochs):
        for record in data_list:
            all_values = record.split(',')
            # print(all_values)
            # image_array = np.asfarray(all_values[1:]).reshape((28,28))
            scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # print(scaled_input)
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(scaled_input, targets)
    file = open('result.txt', 'w')
    res = n.get()
    #print(len(res[5][0]))
    # return [self.inodes,  self.hidnodes, self.outnodes, self.lrn, self.weight_in, self.weight_out]
    for i in range(4):
        file.write(str(res[i])+'\n')
    for i in res[4]:
        for x in i:
            file.write(str("%.5f" % (x)) + ' ')
        file.write('\n')
        #file.write('\n')
    for i in res[5]:
        for x in i:
            file.write(str(x) + ' ')
        file.write('\n')
    file.close()
    print (len(res[4]), len(res[4][0]))
    print (len(res[5]), len(res[5][0]))

def read():
    file = open('result.txt', 'r').readlines()
    input_nodes  = int(file[0])
    hidden_nodes  = int(file[1])
    output_nodes = int(file[2])
    training_rate = float(file[3])

    weight_in = np.zeros((100, 784))
    weight_out = np.zeros((10, 100))
    for i in range(4,104):
        weight_in[i-4] = np.asfarray(file[i][:-2].split(' '))
    for i in range(105, 114):
        weight_out = np.asfarray(file[i-105][:-2].split(' '))
    n.set(input_nodes,  hidden_nodes, output_nodes,training_rate, weight_in, weight_out)
    #надо изменить формат матрицы весов входа и выхода, чтобы скалярное произвидение было ок

def start():
    read()
    check()

'''

input_nodes = 784
output_nodes = 10
hidden_nodes = 100
learning_rate = 0.3
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
data_file = open("mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()
epochs = 2
for e in range(epochs):
    for record in data_list:
        all_values = record.split(',')
        # print(all_values)
        # image_array = np.asfarray(all_values[1:]).reshape((28,28))
        scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # print(scaled_input)
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(scaled_input, targets)

test_data_file = open('mnist_train.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
score = 0
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if label == correct_label:
        score += 1
print(score/(1.0*len(test_data_list)))
check()
