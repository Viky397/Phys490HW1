# Importing the necessary packages

import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import random

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

    # Writing the data to a txt file
    out_file = open("1_test.out","w+")
    for i in w_star:
        out_file.write(str(i))
        out_file.write("\n")
    out_file.write("\n")
    
# A function to compute the solution from stochastic gradient descent 

def gradient_descent(x, y):

    with open(args.input_json) as input_json:
        parsed_json = json.load(input_json)
        learning_rate = parsed_json["learning rate"]
        num_iter = parsed_json["num iter"]
        # print(learning_rate, num_iter)
    
    w = np.zeros((np.shape(x)[1]))

    for i in range(num_iter):

        rand_row = random.randint(0, (len(x)-1))

        derivative_J = ((w.T @ x[rand_row,:])- y[rand_row]) * x[rand_row,:]
        w = w - learning_rate * derivative_J
        print(w)

    out_file = open("1_test.out")
    for i in w:
        out_file.write(str(i))
        out_file.write("\n")
   


if __name__ == "__main__":
    
    args = argparser()

    input_file = np.loadtxt(args.input_file)
    x_vals, y = input_file[:,:-1], input_file[:, -1]
    x_constant = np.ones((len(y), 1))
    x = np.column_stack((x_constant, x_vals))

    analytic(x, y)
    gradient_descent(x, y)