#-----------------------------------
# Author:      Rudiger von Hackewitz 
#-----------------------------------
# Code for the Final Project Report
# Due Date:    Friday, 15 June 2018 

import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL     import ImageTk
from PIL     import Image
from os      import path
from math    import log 
from support import read_ini_section, read_ini_parameter
from kmeans  import KMeansImage 
from dp      import DPImage 


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        # Frame with header information 
        self.header_frame = tk.Frame(master)
        self.header_frame.pack(side="top")
        
        # Frame with source image information 
        self.srcm_frame = tk.Frame(master,highlightcolor="green",highlightbackground="green",highlightthickness=2, bd=0)
        self.srcm_frame.pack(side="top")
        
        # Frame with control items  
        self.srcimg_control = tk.Frame(self.srcm_frame)
        self.srcimg_control.pack( side = "top" )
        
        # Frame for display of source images (original vs cubed image) 
        self.srcimg_frame = tk.Frame(self.srcm_frame)
        self.srcimg_frame.pack( side = "top" )
        
        self.srcorig_frame = tk.Frame(self.srcimg_frame)
        self.srcorig_frame.pack( side = "left" )
        
        self.srccube_frame = tk.Frame(self.srcimg_frame)
        self.srccube_frame.pack( side = "right" )
            
        # Frame with control items  
        self.clustm_control = tk.Frame(master,highlightcolor="blue",highlightbackground="blue",highlightthickness=2, bd=0)
        self.clustm_control.pack( side = "top",padx=10, pady=10)
          
        # Frame with control items  
        self.srcimg2_control = tk.Frame(self.clustm_control)
        self.srcimg2_control.pack( side = "top" )
        
        # Frames to run and display results for KMEANS and DP Clustering algorithms  
        self.cluster_frame = tk.Frame(self.clustm_control)
        self.cluster_frame.pack( side = "top" )
        
        self.dp_frame = tk.Frame(self.cluster_frame)
        self.dp_frame.pack( side = "left" )
        
        self.kmeams_frame = tk.Frame(self.cluster_frame)
        self.kmeams_frame.pack( side = "right" )
        
        # control frames for KMEANS 
        self.kmeams_head = tk.Frame(self.kmeams_frame)
        self.kmeams_head.pack( side = "top" )
        
        self.kmeams_img_frame = tk.Frame(self.kmeams_frame)
        self.kmeams_img_frame.pack( side = "top" )
        
        self.kmeams_ctrl = tk.Frame(self.kmeams_frame)
        self.kmeams_ctrl.pack( side = "top" )
        
        self.kmeams_out = tk.Frame(self.kmeams_frame)
        self.kmeams_out.pack( side = "top" )
        
        # control frames for DP 
        self.dp_head = tk.Frame(self.dp_frame)
        self.dp_head.pack( side = "top" )
        
        self.dp_img_frame = tk.Frame(self.dp_frame)
        self.dp_img_frame.pack( side = "top" )
        
        self.dp_ctrl = tk.Frame(self.dp_frame)
        self.dp_ctrl.pack( side = "top" )
        
        self.dp_out = tk.Frame(self.dp_frame)
        self.dp_out.pack( side = "top" )
        
        
        self.create_widgets()

    def create_widgets(self):
        # populate the title frame with details 
        #######################################
        self.myTitle1 = tk.Label(self.header_frame,text="Image Clustering - CSC8004 (Data Mining)")
        self.myTitle1.pack(side="top")
        self.myTitle2 = tk.Label(self.header_frame,text="Semester 1 2018, June 2018")
        self.myTitle2.pack(side="top")
        self.myTitle3 = tk.Label(self.header_frame,text="Author: Rudiger von Hackewitz")
        self.myTitle3.pack(side="top")
        self.myTitle4 = tk.Label(self.header_frame,text="Student ID: U1088723")
        self.myTitle4.pack(side="top")
    
        # frame for the Source Image information
        ########################################
               
        # source image load button 
        self.load_pic = tk.Button(self.srcimg_control)
        self.load_pic["text"] = "Load  Source Picture"
        self.load_pic["command"] = self.load_it
        self.load_pic.pack(side = "left")    
        
        
        # dropdown list for RGB Cube Length 
        sc = read_ini_section('GLOBAL') 
        self.granularity = int(read_ini_parameter(sc,'Granularity'))
        coptions = ["RGB Cube Length:  "+str(pow(2,i)) for i in range(int(log(self.granularity,2)),7)]
        self.c_value = tk.StringVar(self.srcimg_control)
        self.c_value.set(coptions[0]) # set the value for granularity in the ini file as the default value 
        self.c_dropdown = tk.OptionMenu(self.srcimg_control, self.c_value, *coptions)
        self.c_dropdown.pack(side = "left")    
        
       
        # dropdown list for distance metric 
        doptions = ["Supremum Distance", "Euclidean Distance", "Manhattan Distance"]
        self.d_value = tk.StringVar(self.srcimg2_control)
        self.d_value.set(doptions[1]) # set default value to Euclidean distance
        self.d_dropdown = tk.OptionMenu(self.srcimg2_control, self.d_value, *doptions)
        self.d_dropdown.pack(side = "left")
        
        # check button whether to log clustered images in log folder 
        self.logging = tk.IntVar()
        self.checkbutton = tk.Checkbutton(self.srcimg2_control,text="Log Segmented Images",
                                          variable=self.logging,
                                          command=self.on_logging_click)
        self.checkbutton.pack(side = "left")
        
        
        
        # read in the global parameter for the distance metric and granularity to be used for preprocessing 
        sc = read_ini_section('GLOBAL') 
        self.dist_metric = read_ini_parameter(sc,'DistanceMetric')
        self.granularity = int(read_ini_parameter(sc,'Granularity'))
        # this data is read in from the app.ini file - APP section 
        sc = read_ini_section('APP') 
        self.iimm = read_ini_parameter(sc,'ImageAtStartup')
        self.logpath = read_ini_parameter(sc,'ImageLogPath')
        
     
        
        # display the original image and the cubed image 
        #################################################
        
        # the first panel will store the original image
        self.image = Image.open(self.iimm)
        # rescale image for display in frame 
        img = self.image_rescaler(self.image)
        self.panelA = tk.Label(self.srcorig_frame,image=img)
        self.panelA.image = img
        self.panelA.pack(side="top", padx=10, pady=10)
        
        # details about first image
        self.origimgl = tk.Label(self.srcorig_frame,text="Image not yet loaded")
        self.origimgl.pack(side = "top")
        
    
        # the second panel will store the cubed image
        self.image2 = Image.open(self.iimm)
        # rescale image for display in frame 
        img2 = self.image_rescaler(self.image2)
        self.panelB = tk.Label(self.srccube_frame,image=img2)
        self.panelB.image = img2
        self.panelB.pack(side="top", padx=10, pady=10)
        
        # display the cube dimensions
        self.CubeSide = tk.Label(self.srccube_frame,text=
                                 "Image not yet loaded")
        self.CubeSide.pack(side = "top")
        
        
        
        # clustered pictures 
        #####################
        
        
        # the first panel will store the means clustered image
        self.kmlabel = tk.Label(self.kmeams_head,text="KMeans Clustering")
        self.kmlabel.pack(side = "top")
        
        self.image3 = Image.open(self.iimm)
        # rescale image for display in frame 
        img3 = self.image_rescaler(self.image3)
        self.panelC = tk.Label(self.kmeams_img_frame,image=img3)
        self.panelC.image3 = img3
        self.panelC.pack(side="top", padx=10, pady=10)
        
        # dropdown list for number of clusters in KMEANS 
        options = [str(i)+" Clusters" for i in range(1,11)]
        options[0] = options[0][:-1] # get rid of the s for 1 
        self.k_value = tk.StringVar(self.kmeams_ctrl)
        self.k_value.set(options[2]) # set default value to k=3 for KMEANS 

        self.k_dropdown = tk.OptionMenu(self.kmeams_ctrl, self.k_value, *options)
        self.k_dropdown.pack(side = "left")

        
        # Button to run Kmeans 
        self.kmean_pic = tk.Button(self.kmeams_ctrl)
        self.kmean_pic["text"] = "KMEANS Segmentation"
        self.kmean_pic["command"] = self.kmean_it
        self.kmean_pic.pack(side = "right")
        
        self.kmeanO = tk.Label(self.kmeams_out,text="KMeans Output")
        self.kmeanO.pack(side = "left")
        
    
        # the second panel will store the dp clustered image
        self.dplabel = tk.Label(self.dp_head,text="Density Peak (DP) Clustering")
        self.dplabel.pack(side = "top")
        
        self.image4 = Image.open(self.iimm)
        # rescale image for display in frame 
        img4 = self.image_rescaler(self.image4)
        self.panelD = tk.Label(self.dp_img_frame,image=img4)
        self.panelD.image = img4
        self.panelD.pack(side="top", padx=10, pady=10)
        
        self.dp_pic = tk.Button(self.dp_ctrl)
        self.dp_pic["text"] = "DP Segmentation"
        self.dp_pic["command"] = self.dp_it
        self.dp_pic.pack(side = "top")
     
    
        self.dpO = tk.Label(self.dp_out,text="Density Peak (DP) Output")
        self.dpO.pack(side = "left")
    
    def image_rescaler(self, image): 
        # this data is read in from the app.ini file - APP section 
        sc = read_ini_section('APP') 
        fix_height = int(read_ini_parameter(sc,'ImageHeight'))
        # get width and height of the image 
        [imageSizeWidth, imageSizeHeight] = image.size
        # keep the proportions for width and height during rescaling 
        scaler = fix_height / imageSizeHeight
        newImageSizeHeight = int(imageSizeHeight*scaler)
        newImageSizeWidth  = int(imageSizeWidth*scaler)
        # rescale image for display in frame 
        image = image.resize((newImageSizeWidth, newImageSizeHeight), Image.ANTIALIAS)
        return ImageTk.PhotoImage(image)
    
    
    
    def on_logging_click(self):
        if self.logging.get() == 1:
            # this data is read in from the app.ini file - APP section 
            sc = read_ini_section('APP') 
            self.logpath = read_ini_parameter(sc,'ImageLogPath')
            messagebox.showinfo("Alert", 
                 "All images, clustered created through this interface, will be stored in directory "+self.logpath)
        else:
            messagebox.showinfo("Alert", 
                 "Clustered images will no longer be recorded in directory "+self.logpath)

        
    def load_it(self):
        # this data is read in from the app.ini file - APP section 
        sc = read_ini_section('APP') 
        fpath = read_ini_parameter(sc,'ImageSourcePath')
        self.iimm = filedialog.askopenfilename(initialdir= path.dirname(fpath))
        
        # ensure a file path was selected
        if len(self.iimm) > 0:
            # the first panel will store the original image
            self.image = Image.open(self.iimm)
            # display image dimensions
            [width, height] = self.image.size
            self.origimgl.configure(text="Source Image, name: "+ path.basename(self.iimm)
                                    +"\r Dimensions: " + str(width) + "x"+ str(height)
                                    +"\r Number of pixels: "+str(width* height))
            # rescale image for display in frame 
            img = self.image_rescaler(self.image)
            self.panelA.configure(image=img)
            self.panelA.image=img   
            # the second panel will store the cubed image
            self.image2 = Image.open(self.iimm)
            dp = DPImage(self.d_value.get()[:-9]) 
            dp.GRANULARITY = int(self.c_value.get()[-3:]) # get the integer portion of the label 
            dp.pre_process_img(self.image)
            self.image2.putdata(dp.get_pre_processed_data())
            # rescale image for display in frame 
            img2 = self.image_rescaler(self.image2)
            self.panelB.configure(image=img2)
            self.panelB.image=img2
        
            # display the cube dimensions
            self.CubeSide.configure(text="Preprocessed Image \r "+self.c_value.get()+"\r " + "Number of distinct pixels: " 
                                    + str(len(dp.pnts.keys())))
      
    def kmean_it(self):   
        # number of clusters in kmeans run  
        k = int(self.k_value.get()[:2]) # cut the first two characters in the string and convert to int
        # run the KMEANS algorithm
        km = KMeansImage(k, self.d_value.get()[:-9], True)
        kmimg = Image.open(self.iimm)
        km.GRANULARITY = int(self.c_value.get()[-3:]) # get the integer portion of the label 
        km.run_img(self.image2)
        pixels = km.get_data_img()
        kmimg.putdata(pixels)
        # log the segmented file 
        if self.logging.get() == 1:
            self.log_image(kmimg, "kmean", k)
        # rescale image for display in panelC 
        kmout = self.image_rescaler(kmimg)
        self.panelC.configure(image=kmout)
        self.panelC.image=kmout
       
        self.kmeanO.configure(text="k: " + str(k)+"\r Number of loops: "+str(km.counter)
        +"\r Runtime (in seconds): "+str(km.seconds))
        
        
    def dp_it(self):   
        dp = DPImage(self.d_value.get()[:-9])
        dpimg = Image.open(self.iimm)
        dp.GRANULARITY = int(self.c_value.get()[-3:]) # get the integer portion of the label 
        dp.run_img(self.image2)
        dpimg.putdata(dp.get_data())  
        # log the segmented file 
        if self.logging.get() == 1:
            self.log_image(dpimg, "dp", len(dp.centroids))
        # rescale image for display in panelD 
        dpout = self.image_rescaler(dpimg)
        self.panelD.configure(image=dpout)
        self.panelD.image=dpout      
       
        self.dpO.configure(text="k: " + str(len(dp.centroids)) 
        + "\r Percentage DC to max pixel distance: "+str(round(100*dp.dc/dp.max_dist,1))+"%"
        + "\r Runtime (in seconds): "+str(dp.seconds))       
        
    def log_image(self, imgg, alg, k):
        fff = path.basename(self.iimm)
        imgg.save(self.logpath+fff+"_"+alg+"_"+str(k)+"_"+self.c_value.get()[-3:].strip()+"_"+self.d_value.get()[:-9]+".jpeg")

# run the GUI application 
def run_gui ():
    root = tk.Tk()
    root.title("KMEANS and Density Peak (DP) Clustering of Images")
    # this data is read in from the app.ini file - APP section 
    sc = read_ini_section('APP') 
    window_size = read_ini_parameter(sc,'WindowSize')
    root.geometry(window_size)  # set windows size 
    app = Application(master=root)
    app.mainloop()
    
    
run_gui() 
