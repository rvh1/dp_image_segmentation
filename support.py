#-----------------------------------
# Author:      Rudiger von Hackewitz 
# Code for the Final Project Report
# Due Date:    Friday, 15 June 2018 

from math      import pow 
import configparser

def read_ini_section (section): 
    config = configparser.ConfigParser()
    config.read('app.ini')
    return config[section] # read the section and return data 

def read_ini_parameter (section, par): 
    # return the parameters for further use 
    return section[par]


# set up the function for the minkowski distance, which is later used for calculation 
# of Euclidean and Manhattan distance
def dist_minkowski(point1, point2, p):
    return pow(sum([pow(abs(point1 - point2),p) for point1, point2 in zip(point1,point2)]), 1/p)

# calculate the distance between two points 
# choose different function for Euclidean, Manhattan and Supremum: 
def dist_func(point1, point2, dist): 
    if dist == 'Euclidean':
        return dist_minkowski(point1, point2, 2)
    elif dist == 'Manhattan':
        return dist_minkowski(point1, point2, 1)
    elif dist == 'Supremum':
        # return dist_minkowski(point1, point2, 'infinite') can't be implemented as calculation, but is a limes
        # therefore, a native calculation is used for the 'Supremum' distance 
        return max([abs(point1 - point2) for point1, point2 in zip(point1,point2)])
    else:
        raise Exception('Distance function '+dist+' is not defined in program')
 


