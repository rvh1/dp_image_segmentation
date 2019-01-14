#-----------------------------------
# Author:      Rudiger von Hackewitz 
# Code for the Final Project Report
# Due Date:    Friday, 15 June 2018 

from math     import exp, log
from datetime import datetime

from support import read_ini_section, read_ini_parameter, dist_func 

# we construct a base class for implementation of a generic Density-Peak (DP) clustering algorithm; 
# it can work on any input list with 'points' and is not restricted to images. 
# 'Points' in the list can be provided as tuples or lists; in the case of images those 'points'
# are the 3-dimensional pixels with the RGB components (each between 0 and 255)
class DPPoints:

    def __init__(self, dist = 'Euclidean'):
        
        self.dist = dist                 # distance function that should be applied. Supported options are: 
                                         # 'Euclidean', 'Manhattan' and 'Supremum' 
        self.points          = list()    # list of original pixel stream (passed into the process)  
        
        # this data is read in from the app.ini file - global section 
        sc = read_ini_section('GLOBAL') 
        self.GRANULARITY   = int(read_ini_parameter(sc,'Granularity'))
        
        # this data is read in from the app.ini file - section for DP algorithm 
        sc = read_ini_section('DP') 
        self.D_SCALING   = float(read_ini_parameter(sc,'DensityScaling'))
        self.DG_SCALING  = int  (read_ini_parameter(sc,'DecisionGraphScaling'))
        self.PCT_OUTLIER = float(read_ini_parameter(sc,'OutlierPercentage'))
        self.DENSITY_MIN = float(read_ini_parameter(sc,'DensityMin'))
        
        self.initialise()
        
        
    def initialise (self):  
        self.seconds = 0
        self.p_map           = dict()    # map original pixels to approximated pixels
        self.pnts            = dict()    # dictionary with pre-processed pixels, value contains count for pixel
        self.dst             = dict()    # dictionary of distances associated with each point  
        self.dens            = dict()    # dictionary of densities associated with each point 
        self.centroids       = dict()    # dictionary of centroids, as identified by the DP algorithm 
        self.assigned_group  = dict()    # dictionary, containing points as keys and associated group as value 
        
        self.max_dist        = 0.0       # maximum distance between any two pixels in the image
        self.dc              = 0.0       # scaling factor for density calculation (rho)
        self.dst_threshold   = 0.0       # calculated distance threshold for outliers in Decision Graph 
        self.dens_threshold  = 0.0       # calculated density threshold for outliers in Decision Graph

   
    # This  method returns the clustered data stream for the image for display
    def get_data(self):
        return [self.assigned_group[self.p_map[p]] for p in self.points] 
    
    # preprocessed image data (to improve runtimes of clustering algorithm)
    def get_pre_processed_data (self):
        return [self.p_map[p] for p in self.points]

    
    def pre_process_points (self, data): 
        self.points = data
        self.pnts = dict()
        self.p_map = dict() 
        for _pp in self.points:
            pp = list(_pp)
            p = tuple(map(lambda x: ((int(x/self.GRANULARITY))*self.GRANULARITY) + int(self.GRANULARITY/2), list(pp)))
            if p not in self.pnts: 
                self.pnts[p] = 1
            else: 
                self.pnts[p] = self.pnts[p] + 1
            if _pp not in self.p_map: 
                self.p_map[_pp] = p
    
    
    # calculate the density for a given point in the list of points: 
    def density (self, p):
        # for points inside the same cube: 
        # use the average distance between two random points in a cube. 
        # The average distance depends on the distance metric:
        if self.dist == 'Euclidean': 
            est = 0.66
        elif self.dist == 'Manhattan':
            est = 1.0 
        else:
            est = 0.54  # Supremum distance
        est_dist = int (self.GRANULARITY * est)
        rho = (self.pnts[p] - 1) * exp(-pow(est_dist/self.dc,2)) 
        # now use the distance between centroids of each of the other cubes to add 
        # to the calculation of the overall density for the point: 
        for pp in self.pnts.keys(): 
            if pp != p: 
                rho = rho + self.pnts[pp]*exp(-pow(dist_func(pp, p, self.dist)/self.dc,2))
        return rho

    # now we use this function to calculate the density of each point in the list; 
    # the function returns a list of densities, with each index corresponding to the 
    # point in the original list that has been passed into the function 
    def density_points (self): 
        # get the maximum distance between any two pixels in the image 
        self.max_dist = max([dist_func(p, q, self.dist) for p in self.pnts.keys() for q in self.pnts.keys()])
        # set density scaling factor
        self.dc = self.max_dist / self.D_SCALING
        # take number of records into consideration in the order of natural logarithm: 
        self.dc = log(float(len(self.points))) + self.dc 
        for p in self.pnts.keys(): 
            self.dens [p] = self.density (p)
        # now scale the density values between 0 and DG_SCALING: 
        m = self.DG_SCALING/max(self.dens.values())
        for p in self.dens:
            self.dens[p] = self.dens[p]*m
        return self.dens
    

    # For each point calculate the minimum distance of the point to another point with higher or equal density: 
    def distance_points (self): 
        pmax = max([p for p in self.pnts.keys()], key = lambda x : self.dens[x])
        self.dst[pmax] = max([dist_func(pmax, pp, self.dist) for pp in self.pnts.keys()])
        for p in self.pnts.keys(): 
            if p != pmax:
                self.dst[p] = min([dist_func(p, pp, self.dist) for pp in self.pnts.keys() if self.dens[pp] > self.dens[p]])    
        # scale the distance value between 0 and DG_sCALING  
        m = self.DG_SCALING/max(self.dst.values())
        for p in self.dst:
            self.dst[p] = self.dst[p]*m       
        return self.dst
    
    
    # get distance threshold for outliers 
    def get_dst_threshold (self): 
        # calculate lambda as the reciprocal value of the emperical mean of the distance values for each record.
        # based on the model that distance follows an exponential distribution 
        lambd = len(self.dst.keys()) / sum(self.dst.values()) 
        self.dst_threshold = -log(self.PCT_OUTLIER)/lambd  
        return self.dst_threshold + log(len(self.pnts.keys()))
    
    
    # get density threshold for outliers 
    def get_dens_threshold (self):  
        return self.DENSITY_MIN*self.DG_SCALING
    
    # get outliers (indexes for points in the list of points): 
    def get_outliers (self):
        self.dst_threshold = self.get_dst_threshold ()
        self.dens_threshold = self.get_dens_threshold ()
        self.centroids = [p for p in self.pnts.keys() 
                              if self.dst[p] > self.dst_threshold and self.dens[p] > self.dens_threshold]
        return self.centroids
    

    # now assign the remaining points, by building up the dictionary assigned_group in recursive calls:
    def assign_point(self, p): 
        if p not in self.assigned_group: 
            # if not yet assigned, then find the closest point with higher density: 
            q = min([pp for pp in self.pnts.keys() 
                         if self.dens[pp] > self.dens[p]], key = lambda x : dist_func(x, p, self.dist))
            # call the same function recursively to find appropriate group for this point: 
            self.assigned_group[p] = self.assign_point(q)
        # return the appropriate pixel centroid for the point 
        return self.assigned_group[p]

    
    # assign points to relevant groups by calling function assign_group for each key in the dictionary  
    def assign_remaining_points (self):
        # first initialise the dictionary with the outliers. They are centroids of their respective clusters
        for p in self.centroids:
            self.assigned_group[p] = p
        # then assign each remaining point in the dictionary to their appropriate group 
        for p in self.pnts.keys():
            self.assign_point(p)
    
    # last not least, run all steps of the DP algorithm in the single function call 'run' and measure the time 
    def run(self, data):
        self.points = data
        # initialise data 
        self.initialise()
        t1 = datetime.now()
        # preprocess the pixels 
        self.pre_process_points(data)
        # assign density and distance for each point: 
        self.dens = self.density_points()
        self.dst  = self.distance_points() 
        # now get the outliers in the decision graph  
        self.centroids = self.get_outliers ()
        # assigne the remaining points to their appropriate groups 
        self.assign_remaining_points()
        t2 = datetime.now()
        self.seconds = (t2-t1).seconds
        

        
from PIL import Image    
# https://pillow.readthedocs.io/en/5.1.x/reference/Image.html 

class DPImage(DPPoints):
    def __init__(self, dist = 'Euclidean'):
        DPPoints.__init__(self, dist)
        self.image = list()
              
    def pre_process_img (self, image): 
        self.image = image
        self.points = list(image.getdata())
        DPPoints.pre_process_points(self, self.points)
        
    def run_img(self, image):
        self.image = image
        self.points = list(image.getdata())
        DPPoints.run(self, self.points)

    
