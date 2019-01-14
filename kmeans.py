#-----------------------------------
# Author:      Rudiger von Hackewitz 
# Code for the Final Project Report
# Due Date:    Friday, 15 June 2018 



import random
from functools import reduce 
from datetime import datetime

from support import read_ini_section, read_ini_parameter, dist_func 

class KMeansPoints:

    def __init__(self, k, dist = 'Euclidean', preprocessing = True):
        
        self.k = k            # input parameter K: number of clusters that should be calculated by algorithm
        
        self.dist = dist                 # distance function that should be applied. Supported options are: 
                                         # 'Euclidean', 'Manhattan' and 'Supremum' 
         
        self.preprocessing   = preprocessing 
        self.points          = list()    # list of original pixel stream (passed into the process) 
        
        # this data is read in from the app.ini file - global section 
        sc = read_ini_section('GLOBAL') 
        self.GRANULARITY   = int(read_ini_parameter(sc,'Granularity'))
        
        # this data is read in from the app.ini file - section for KMEANS algorithm 
        sc = read_ini_section('KMEANS') 
        self.SET_RANDOM_SEED   = bool(read_ini_parameter(sc,'SetRandomSeed') == 'yes')
        
        self.initialise()
        
        
    def initialise (self):  
        self.seconds         = 0         # measure time for run
        self.counter         = 0         # record the number of loops in the KMEANS process 
        self.p_map           = dict()    # map original pixels to approximated pixels
        self.pnts            = dict()    # dictionary with pre-processed pixels, value contains count for pixel 
        self.assigned_group  = dict()    # dictionary with assigned groups
        self.prnt_sum        = dict() 
        self.means           = list()    # current means values 
        self.new_means       = list()    # newly calculate means values 
        self.prnt_sum        = dict() 
   
    # This  method returns the clustered data stream 
    def get_data(self):
        return [[p for p in self.points if self.assigned_group[p] == i] for i in range(self.k)]
    
    # preprocessed image data (to improve runtimes of clustering algorithm)
    def get_pre_processed_data (self):
        return [self.p_map[p] for p in self.points]

    # pre-process the image data 
    def pre_process_points (self, data): 
        self.points = [tuple(p) for p in data] 
        self.pnts = dict()
        self.p_map = dict() 
        for _pp in self.points:
            pp = list(_pp)
            if self.preprocessing:
                p = tuple(map(lambda x: ((int(x/self.GRANULARITY))*self.GRANULARITY) + int(self.GRANULARITY/2), list(pp)))
            else:
                p = tuple(pp)
            if p not in self.pnts: 
                self.pnts[p] = 1
            else: 
                self.pnts[p] = self.pnts[p] + 1
            if _pp not in self.p_map: 
                self.p_map[_pp] = p
    
            
    # adds the values of two lists for each dimension    
    # for example:  sum_dimensions([1,2,3],[4,5,6]) = [5,7,9]
    def sum_dimensions(self,lst1,lst2):
        return [lst1_i + lst2_i for lst1_i, lst2_i in zip(lst1, lst2)]


    # calculate the mean value of the lists across all dimensions 
    # (which are the 3 RGB channels for images)
    # input is a list of points (lists)
    # for example: mean_func([[1,2,3],[4,5,6]]) = [2.5, 3.5, 4.5]
    def mean_func(self,lists, i): 
        l = sum([self.pnts[tuple(p)] for p in self.pnts if self.assigned_group[p] == i])
        if l > 0:   # avoid division by zero 
            # sum up across all dimensions 
            vs = reduce(self.sum_dimensions, lists)
            # divide by the number of points to get the mean across each dimension
            return [(1/l) * v_i for v_i in vs]
        else:
            return 0; # consider the mean to be zero when the list is empty

    # last not least, run all steps of the DP algorithm in the single function call 'run' and measure the time 
    def run(self, data):
        self.points = [tuple(p) for p in data]
        # initialise data 
        self.initialise()
        t1 = datetime.now()
        # preprocess the pixels 
        self.pre_process_points(data)
        # pick k random points to start with the process and assign points to groups accordingly
        if self.SET_RANDOM_SEED:
            random.seed(self.k * len(self.points))
        rnd  = random.sample(self.pnts.keys(), self.k) 
        self.new_means = [list(r) for r in rnd]
        self.means = list()  
        self.prnt_sum = dict()

        # assign groups accordingly to the centre they are closest to 
        for p in self.pnts.keys():
            self.assigned_group[p] = min([i for i, _ in enumerate(self.new_means)], 
                                         key = lambda x: dist_func(list(p), list(self.new_means[x]), self.dist))  
            self.prnt_sum [p] = list([i * self.pnts[p] for i in list(p)]) 
        
        # break, once we don't get any changes any more in the cluster means 
        while self.means != self.new_means:
            self.counter = self.counter + 1         # increment counter by 1 
            self.means = self.new_means
            # Recalculate the k centroids, based on the new data
            self.new_means = [self.mean_func([self.prnt_sum [p] for p in self.pnts.keys() if self.assigned_group[p] == i], i) 
                              for i in range(self.k)]
            
            # assign groups accordingly to the centre they are closest to 
            for p in self.pnts.keys():
                self.assigned_group[p] = min([i for i, _ in enumerate(self.new_means)], 
                                             key = lambda x: dist_func(list(p), list(self.new_means[x]), self.dist)) 
        t2 = datetime.now()
        self.seconds = (t2-t1).seconds

#
# now derive a separate class for the kmeans algorithm on images. It is basically 'wrapped around'
# the base class KMeansPoints and adds some specific features for the processing of pixels (as 
# some specialised kinds of 'points'. In particular: pixels need to be returned in the correct order
# and the means of pixels need to be integers (rather than floats). 
        
from PIL import Image    
# https://pillow.readthedocs.io/en/5.1.x/reference/Image.html 

class KMeansImage(KMeansPoints): 
    def __init__(self, k, dist='Euclidean',preprocessing=True):
        KMeansPoints.__init__(self, k, dist, preprocessing)
        self.image = list()
              
    def pre_process_img (self, image): 
        self.image = image
        self.points = list(image.getdata())
        KMeansPoints.pre_process_points(self, self.points)
        
    def run_img(self, image):
        self.image = image
        self.points = list(image.getdata())
        KMeansPoints.run(self, self.points)
    def get_data_img (self):
        # now convert to an integer
        rmeans = [list(map(int,m)) for m in self.means]
        return [tuple(rmeans[self.assigned_group[self.p_map[p]]]) for p in self.points] 