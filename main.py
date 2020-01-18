# Importing the necessary packages

import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import random
import os

# A function to create command line flags

def argparser():
    
    parser = argparse.ArgumentParser(
        description="490_HW1")
    parser.add_argument("--input_file",
                        help="input file")
    parser.add_argument("--input_json",
                        help="input json")
    args = parser.parse_args()

    return args

# A function to compute the analytic solution to the least square regression 

def analytic(x, y):

    w_star = np.around(np.dot((np.linalg.inv(np.dot(x.T, x))), np.dot(x.T, y)), decimals=1)

    # Writing the data to a .txt file
    out_file = open(output,"w+")
    for i in w_star:
        out_file.write(str(i))
        out_file.write("\n")
    out_file.write("\n")
    
# A function to compute the solution from stochastic gradient descent 

def gradient_descent(x, y):

    # Extracting the learning rate and number of iterations from .json
    with open(args.input_json) as input_json:
        parsed_json = json.load(input_json)
        learning_rate = parsed_json["learning rate"]
        num_iter = parsed_json["num iter"]
    
    # Declare an array of zeros, with the same number of rows as 'x'
    w = np.zeros((np.shape(x)[1]))

    for i in range(num_iter):

        # Choosing a random row from 'x'
        rand_row = random.randint(0, (len(x)-1))

        # '@' is matrix multiplication
        derivative_J = ((w.T @ x[rand_row,:])- y[rand_row]) * x[rand_row,:]
        w = w - learning_rate * derivative_J
    
    # Appending the data to the existing .txt file
    out_file = open(output, 'a')
    for i in w:
        out_file.write(str(i))
        out_file.write("\n")
   
if __name__ == "__main__":
    
    args = argparser()

    input_file = np.loadtxt(args.input_file)
    x_vals, y = input_file[:,:-1], input_file[:, -1]
    output = str((((os.path.split(args.input_file))[1]).split('.'))[0]) + ".out"
 
    # Adding the bias column of ones to 'x'
    x_constant = np.ones((len(y), 1))
    x = np.column_stack((x_constant, x_vals))

    analytic(x, y)
    gradient_descent(x, y)