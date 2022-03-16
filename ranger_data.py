import os
import glob
import numpy as np
from copy import copy, deepcopy
from PIL import Image
from tqdm import *
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# create a ranger data object
class ranger():
    def __init__(self,rpath=None, RS=0, buildup=0, baselines=None, calibration=None):
        '''
        Initialise a Ranger data object with the following input parameters:
            rpath           - Directory MUST contain Ranger data (.bmp files and activescript.txt)
            RS              - Range shifter WET (mm)
            buildup         - Ranger buildup insert WET (mm)
            calibration     - Parameters for analytical conversion of IDDs from pixels to mm (WET)
            baselines       - Giraffe data used to produce calibration parameters
        '''
        self.rpath = rpath
        self.RS = RS
        self.buildup = buildup
        self.calibration={
            'pixel_pitch': 3.60289513376288,
            'scintillator_WER': 0.937915407838645,
            'window_WER': 1.40479799993566,
            'window_thickness': 8.44546391423171,
            'pixel_offset': 3.57054925419925,
        }
        if calibration:
            for key in calibration.keys():
                self.calibration[key]=calibration[key]

        if baselines:
            self.baselines=baselines
        else:
            self.baselines={
                'Energy':[210,
                          200,
                          190,
                          180,
                          170,
                          160,
                          150,
                          140,
                          130,
                          120,
                          110,
                          100,
                          90,
                          80,
                          70 ],
                'D80':[282.1,
                       259.5,
                       237.5,
                       216.2,
                       196.0,
                       176.5,
                       157.8,
                       139.9,
                       122.7,
                       106.5,
                       91.5,
                       77.4,
                       64.0,
                       51.7,
                       41.0]
            }

    
    def load_data(self,rpath=None):
        ''' load Ranger bmp images to obtain IDDs'''
        if not rpath:
            rpath = self.rpath
        else:
            self.rpath = rpath

        bmp_list = glob.glob(os.path.join(rpath,'*.bmp'))
        img0 = Image.open(bmp_list[0])
        img0 = np.array(img0)
        self.img = np.empty((len(bmp_list),img0.shape[0],img0.shape[1]))
        self.img[0,:] = img0
        for i in (pbar := tqdm(range(1,len(bmp_list)))):
            self.img[i,:] = np.array(Image.open(bmp_list[i]))
            pbar.set_description('Loading '+os.path.basename(bmp_list[i]))
        saturation_mask = self.img>=255
        self.saturated_pixels = np.count_nonzero(saturation_mask)
        
        with open(os.path.join(rpath,'activescript.txt')) as txt_info:
            activescript = txt_info.read()
            idx = activescript.index('CameraHRatio=  ')
            idx = idx+15
            self.hratio = float(activescript[idx:idx+6])
            idx = activescript.index('CameraVRatio=  ')
            idx = idx+15
            self.vratio = float(activescript[idx:idx+6])
        
        with open(os.path.join(rpath,'activescript.txt')) as txt_info:
            lines = txt_info.readlines()
            c = 0
            for l in lines:
                if 'ROI_top=' in l:
                    self.roi_top = float(l[8:])
                    c+=1
                if 'ROI_left=' in l:
                    self.roi_left = float(l[9:])
                    c+=1
                if 'ROI_width=' in l:
                    self.roi_width = float(l[10:])
                    c+=1
                if 'ROI_bottom=' in l:
                    self.roi_bottom = float(l[11:])
                    c+=1
                if 'AppXCenter=' in l:
                    self.x_centre = float(l[11:])
                    c+=1
                if 'AppYCenter=' in l:
                    self.y_centre = float(l[11:])
                    c+=1
                if c == 6:
                    break
        
        with open(os.path.join(rpath,'winlvsapp.ini')) as ini_info:
            lines = ini_info.readlines()
            c = 0
            for l in lines:
                if 'VideoFrameRate=' in l:
                    self.fps = float(l[15:])
                    c+=1
                if 'VideoShutter=' in l:
                    self.shutter = float(l[13:])
                    c+=1
                if 'VideoGain=' in l:
                    self.gain = float(l[10:])
                    c+=1
                if c == 3:
                    break
        
        self.idd() # generate IDDs
        self.idd_metrics() # analyse IDDs

    def plot_img(self,i):
        ''' plot a bmp image (useful for checking orientation)'''
        img = np.squeeze(self.img[i,:])
        plt.imshow(img, cmap='gray')
        plt.plot([self.x_centre,self.x_centre],[self.y_centre-50,self.y_centre+50],'m')
        plt.plot([self.x_centre-50,self.x_centre+50],[self.y_centre,self.y_centre],'m')
        plt.plot(
            [self.roi_left,self.roi_left+self.roi_width,self.roi_left+self.roi_width,self.roi_left+self.roi_width,self.roi_left,self.roi_left],
            [self.roi_top,self.roi_top,self.roi_bottom,self.roi_bottom,self.roi_bottom,self.roi_top],
            'r')
        plt.show

    def idd(self):
        ''' produce idd by integrating over ROI (NB: beam must originate from bottom of image!)'''
        self.raw_idd = []
        r_min = int(self.roi_top)
        r_max = int(self.roi_bottom)
        c_min = int(self.roi_left)
        c_max = int(self.roi_left+self.roi_width)
        stack_crop = self.img[:,r_min:r_max,c_min:c_max]
        for i in range(stack_crop.shape[0]):
            img = np.squeeze(stack_crop[i,:])
            raw_idd = np.sum(img,axis=-1)
            raw_idd = raw_idd / raw_idd.max()
            self.raw_idd.append(raw_idd)

    def plot_idd(self,i, E=None,axis=-1):
        idd = copy(self.raw_idd[i])
        if axis == -1:
            x = np.array(range(idd.shape[0]*-1,0))*-1
        else:
            x = np.array(range(idd.shape[0]))
        plt.plot(x,idd)
        plt.xlabel('Depth (px)')
        plt.ylabel('Dose (norm)')
        if E:
            plt.title(str(E)+' MeV')
        plt.show
    
    def idd2mm(self):
        ''' convert IDDs from pixels to WET mm'''
        px_pitch = self.calibration['pixel_pitch']
        scint_wer = self.calibration['scintillator_WER']
        wind_thickness = self.calibration['window_thickness']
        wind_wer = self.calibration['window_WER']
        px_offset = self.calibration['pixel_offset']
        window_WET = wind_thickness/wind_wer/px_pitch # PTFE window screwed onto ranger
        camera_offset = px_offset/scint_wer # some notional camera offset
        range_offset = window_WET+camera_offset+self.buildup+self.RS # total range offset inlucing WET of RS and PTFE insert
        P100_WET = (np.array(self.metrics['P100'])-wind_thickness)/scint_wer/px_pitch # range in scintillator
        P100_WET = P100_WET+ range_offset
        P80_WET = (np.array(self.metrics['P80'])-wind_thickness)/scint_wer/px_pitch # range in scintillator
        P80_WET = P80_WET+ range_offset
        P90_WET = (np.array(self.metrics['P90'])-wind_thickness)/scint_wer/px_pitch # range in scintillator
        P90_WET = P90_WET+ range_offset
        D80_WET = (np.array(self.metrics['D80'])-wind_thickness)/scint_wer/px_pitch # range in scintillator
        D80_WET = D80_WET+ range_offset
        D90_WET = (np.array(self.metrics['D90'])-wind_thickness)/scint_wer/px_pitch # range in scintillator
        D90_WET = D90_WET+ range_offset
        D20_WET = (np.array(self.metrics['D20'])-wind_thickness)/scint_wer/px_pitch # range in scintillator
        D20_WET = D20_WET+ range_offset
        self.metrics_mm = {'P100':P100_WET.tolist(), 'P80':P80_WET.tolist(), 'P90':P90_WET.tolist(), 'D90':D90_WET.tolist(), 'D80':D80_WET.tolist(), 'D20':D20_WET.tolist()}

    def idd_metrics(self, plot=False):
        ''' extract metrics from IDDs '''
        metrics = {'P100':[], 'P80':[], 'P90':[], 'D90':[], 'D80':[], 'D20':[]}
        peaks = np.empty((len(self.raw_idd),2))
        n=0
        for y in self.raw_idd:
            if plot:
                plt.plot(y)
            idx, peak_metrics = find_peaks(y, width=y.max()*0.1, prominence=y.max()*0.075, height=y.max()*0.2)
            x100 = idx.min()
            x0 = idx.max()
            pk = peak_metrics['peak_heights'].max()
            peaks[n,1] = pk
            peaks[n,0] = x100
            dist = copy(y)
            prox = copy(y)
            dist[x100:]=0
            prox[0:x100]=0
            #Bragg peak (P100)
            metrics['P100'].append(x0 - x100)
            # P80
            x = np.abs(prox - pk*0.8).argmin()
            metrics['P80'].append(x0-x)
            # P90
            x = np.abs(prox - pk*0.9).argmin()
            metrics['P90'].append(x0-x)
            # D90
            x = np.abs(dist - pk*0.9).argmin()
            metrics['D90'].append(x0-x)
            # D80
            x = np.abs(dist - pk*0.8).argmin()
            metrics['D80'].append(x0-x)
            # D20
            x = np.abs(dist - pk*0.2).argmin()
            metrics['D20'].append(x0-x)
            n+=1
        self.metrics=metrics
        self.idd2mm()
        if plot:
            print(peaks)
            plt.plot(peaks[:,0],peaks[:,1],'k|')

    

