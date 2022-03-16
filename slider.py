from openslide import OpenSlide
import xml.etree.ElementTree as ET
import numpy as np
import os, glob
import time
import argparse, sys

class Slider:

    def __init__(self, filename,level, output_folder_PAS, output_folder_HE, display=False):
        self.filename = filename
        self.level = level
        self.display = display
        self.output_folder_PAS = output_folder_PAS
        self.output_folder_HE = output_folder_HE

    def openSlide(self):
        self.slide = OpenSlide(self.filename + ".mrxs")

    def getAllRectangles(self):
        tree = ET.parse(self.filename + '.xml')
        root = tree.getroot()
        self.boxes = []
        for annotation in root.iter('Annotation'):
            if "Rectangle" in annotation.attrib.values():
                x = []
                y = []
                for corr in annotation.iter("Coordinate"):
                    x.append(int(float(corr.get("X"))))
                    y.append(int(float(corr.get("Y"))))
                x1 = min(x) - 300
                y1 = min(y) - 300
                x2 = max(x) + 300  #because dots are out of rectangle
                y2 = max(y) + 300
                self.boxes.append((x1,y1,x2,y2))

    def getAllDots(self):
        tree = ET.parse(self.filename + '.xml')
        root = tree.getroot()
        self.dots = []
        for annotation in root.iter('Annotation'):
            if "Dot" in annotation.attrib.values():
                for corr in annotation.iter("Coordinate"):
                    self.dots.append(((int(float(corr.get("X")))), int(float(corr.get("Y")))))

    
    def assignDot(self,dot):
        for i in range(0,len(self.boxes)):
            if( dot[0] > self.boxes[i][0] and dot[0] < self.boxes[i][2]
            and dot[1] > self.boxes[i][1] and dot[1] < self.boxes[i][3]):
                row = np.array([[dot[0], dot[1],i]])
                self.assignedDots = np.append(self.assignedDots, row, axis=0)
                i +=1

    def assingDotsToBoxes(self):
        self.assignedDots = np.empty((0,3), int)
        for dot in self.dots:
                self.assignDot(dot)
        if(self.assignedDots[:,0].size == len(self.dots)):
            print("All dots have been assigned to rectangles in " + filename )
        else:
            print("Error, not every dot was assigned in " + filename)
            

    def openRegion(self, box):
        size_x = int((box[2]-box[0])/pow(2,self.level)) ## by scaling the photo, only size is changing 
        size_y = int((box[3]-box[1])/pow(2,self.level))
        self.region = self.slide.read_region((box[0],box[1]),self.level,(size_x,size_y))  

    def extractData(self):
        i=0
        for dot in self.assignedDots:
            if(dot[-1] == self.currentBox):
                width = 512-50*self.level   ## how much should image have width and height
                area = (self.assignedDots[i][0]-width-self.boxes[self.currentBox][0], self.assignedDots[i][1]-width-self.boxes[self.currentBox][1], 
                        self.assignedDots[i][0]+width-self.boxes[self.currentBox][0], self.assignedDots[i][1]+width-self.boxes[self.currentBox][1])
                area_scaled = [int(point*pow(0.5,self.level)) for point in area]
                cropped_img = self.region.crop(area_scaled)
                file = self.getNameToSavedImage(i)
                cropped_img.save(file)
            i = i+1

    def getNameToSavedImage(self, numberOfDot):
        if("PAS" in self.filename):
            file = self.output_folder_PAS
        if("HE" in self.filename):
            file = self.output_folder_HE
        file = file + os.path.basename(self.filename) + "_"
        file = file + str(numberOfDot)
        if("PAS" in self.filename):
            file = file + "_PAS_s"
        elif("HE" in self.filename):
            file = file + "_HE_s"
        file = file + str(level) + ".png"
        return file


    def displayInfo(self):
        print("lvl count "  + str(self.slide.level_count))
        print("dimension: " + str(self.slide.dimensions))
        print("lvl 0 dimension" + str(self.slide.level_dimensions[0]))
        print("lvl 1 dimension" + str(self.slide.level_dimensions[1]))
        print("lvl 2 dimension" + str(self.slide.level_dimensions[2]))
        print("lvl 0 downsample" + str(self.slide.level_downsamples[0]))
        print("lvl 1 downsample" + str(self.slide.level_downsamples[1]))
        print("lvl 2 downsample" + str(self.slide.level_downsamples[2]))

    def showStatistics(self):
        print("number of dots = ", len(self.dots))
        print("number of assigned dots = ", len(self.assignedDots))

    def main(self):  
        print("\n-------- STARTING OPERATIONS WITH FILE ", self.filename, "---------")
        self.openSlide()
        self.getAllRectangles()
        self.openRegion(self.boxes[0])


        self.getAllDots()
        self.assingDotsToBoxes()
        for i in range (0,len(self.boxes)):
            self.currentBox = i
            self.openRegion(self.boxes[self.currentBox])
            self.extractData()
        self.showStatistics()
        if self.display:
            self.displayInfo()
        if(len(self.dots) == len(self.assignedDots[:,0])):
            return True
        else:
            return False

if __name__ == '__main__':
    
    
    parser=argparse.ArgumentParser()

    parser.add_argument('--level', help='list of resolutions of slides which should be loaded, range is 0-13, default [0]',default=[0])
    parser.add_argument('--PAS_folder', help='folder that contains PAS slides files with extension .mrxs, default: PAS_przebarwiony/*.mrxs', default='PAS_przebarwiony/*.mrxs')
    parser.add_argument('--HE_folder', help='folder that contains HE slides files with extension .mrxs, default: HE_do_recznego/*.mrxs', default= 'HE_do_recznego/*.mrxs')
    # train images: 'PAS_przebarwiony/*.mrxs' and 'HE_do_recznego/*.mrxs'
    # test images:'HE_do_recznego/testowe/*.mrxs' and 'PAS_przebarwiony/testowe/*.mrxs'
    
    parser.add_argument('--output_folder_PAS', help='localization where extracted PAS images should be located', default="output/PAS_s0/")
    parser.add_argument('--output_folder_HE', help='localization where extracted HE images should be located', default="output/HE_s0/")  
    #"output/PAS_s0/" or "output/PAS_s0/test_PAS/"
    #"output/HE_s0/" or "output/HE_s0/test_HE/"
    
    parser.add_argument('--filename', help='.mrxs file which specifically should be loaded, if given PAS/HE_folder ignored')
    parser.add_argument('--display', help='boolean, show information about slides, number of levels/dimension etc.', default=False)
    
    args = parser.parse_args()
    
    if args.filename: #eg. filename = "PAS_przebarwiony/AA7756"  or filename = "HE_do_recznego/AA7756"  
        slider = Slider(args.filename, args.level)
        slider.main()
        sys.exit()

    start_time = time.time()
    results = np.empty(0)

    files =  glob.glob(args.PAS_folder) + glob.glob(args.HE_folder)
    for file in files:
        filename = os.path.splitext(file)[0]
        for j in args.level:        
            level = j
            slider = Slider(filename, level, args.output_folder_PAS,args.output_folder_HE, args.display)
            results = np.append(results, slider.main())
    print("Final results of assigning dots = ", results)

    print("--- %s seconds ---" % (time.time() - start_time))      

