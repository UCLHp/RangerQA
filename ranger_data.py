import os
import glob
import warnings
import numpy as np
from copy import copy, deepcopy
from PIL import Image
from tqdm import *
from scipy.signal import find_peaks
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Ranger cross-calibration data collected from Giraffe 16/02/2022
# Update calibration data when required by passing a new dict to cal_data when initialising ranger class
calibration={
            'pixel_pitch': 3.60289513376288,
            'scintillator_WER': 0.937915407838645,
            'window_WER': 1.40479799993566,
            'window_thickness': 8.44546391423171,   # window thickness in pixels
            'pixel_offset': 3.57054925419925,
            'buildup': 0,
            'RS': 0,
            'Energy':[      210,
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
                            70],
            'CalD80':[      282.1,
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
                            41.0],
            'RangerD80':[   944,
                            866,
                            793,
                            721,
                            653,
                            585,
                            522,
                            463,
                            404,
                            349,
                            300,
                            252,
                            207,
                            166,
                            127],
            }
# reference IDD data collected from "Giraffe - QA Record.xlsx" 17/3/2022 and WET measurements from 16/02/2022 on Gantry 3
# Update reference data when required by passing a new dict to ref_data when initialising ranger class
reference_data = {
    'Energy': [245, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70], #MeV
    'D20': {'G1':[],'G2':[],'G3':[],'G4':[],'TPS':[]}, # mm
    'D80': {'G1':[],'G2':[],'G3':[],'G4':[],'TPS':[]}, # mm
    'D90': {'G1':[],'G2':[],'G3':[],'G4':[],'TPS':[]}, # mm
    'RS 5cm': 57.36666667, # mm
    'RS 3cm': 34.26666667, # mm
    'RS 2cm': 22.86666667, # mm
    'PMMA 11.05': 13.12222222, # mm
    'PMMA 11.10': 13.09555556, # mm
    'PTFE 13.27': 23.67555556, # mm
}
reference_data['D20']['TPS']=[  372.1197284,
                                359.463304,
                                334.9744326,
                                310.48687,
                                287.0634392,
                                264.3503537,
                                242.2839088,
                                221.0633487,
                                200.3423747,
                                180.4255655,
                                161.3153755,
                                143.0961101,
                                125.6672448,
                                109.2610037,
                                93.91957573,
                                79.51620082,
                                65.8189913,
                                53.16120468,
                                42.14013074,
                                ]  
reference_data['D80']['TPS']=[  366.8390191,
                                354.3505605,
                                329.9325405,
                                305.5543291,
                                282.2033654,
                                259.625619,
                                237.6437232,
                                216.4325858,
                                195.970254,
                                176.3014255,
                                157.4935619,
                                139.5692477,
                                122.4802512,
                                106.3888415,
                                91.26672154,
                                77.13787452,
                                63.86207031,
                                51.57160758,
                                40.77587833,
                                ]   
reference_data['D90']['TPS']=[  365.7403053,
                                353.2687547,
                                328.8676978,
                                304.5187681,
                                281.1972222,
                                258.6413816,
                                236.6531579,
                                215.4729073,
                                195.0420795,
                                175.423794,
                                156.7062278,
                                138.8364286,
                                121.8096748,
                                105.7739059,
                                90.69101083,
                                76.61840125,
                                63.44521951,
                                51.23237633,
                                40.47195652,
                                ]    
reference_data['D20']['G1']=[   372.367,
                                359.550,
                                334.850,
                                310.867,
                                287.117,
                                264.500,
                                242.350,
                                220.783,
                                200.367,
                                180.483,
                                161.367,
                                143.083,
                                125.683,
                                108.967,
                                93.800,
                                79.300,
                                65.850,
                                53.317,
                                42.200,
                                ]
reference_data['D80']['G1']=[   366.667,
                                354.067,
                                329.467,
                                305.580,
                                281.800,
                                259.300,
                                237.217,
                                215.900,
                                195.700,
                                176.167,
                                157.483,
                                139.400,
                                122.400,
                                106.150,
                                91.200,
                                77.000,
                                63.900,
                                51.717,
                                40.817,
                                ]
reference_data['D90']['G1']=[   365.550,
                                352.967,
                                328.367,
                                304.517,
                                280.733,
                                258.233,
                                236.200,
                                214.900,
                                194.767,
                                175.300,
                                156.667,
                                138.700,
                                121.733,
                                105.567,
                                90.650,
                                76.500,
                                63.500,
                                51.400,
                                40.500,
                                ]
reference_data['D20']['G1']=[   372.367,
                                359.550,
                                334.850,
                                310.867,
                                287.117,
                                264.500,
                                242.350,
                                220.783,
                                200.367,
                                180.483,
                                161.367,
                                143.083,
                                125.683,
                                108.967,
                                93.800,
                                79.300,
                                65.850,
                                53.317,
                                42.200,
                                ]
reference_data['D80']['G1']=[   366.667,
                                354.067,
                                329.467,
                                305.580,
                                281.800,
                                259.300,
                                237.217,
                                215.900,
                                195.700,
                                176.167,
                                157.483,
                                139.400,
                                122.400,
                                106.150,
                                91.200,
                                77.000,
                                63.900,
                                51.717,
                                40.817,
                                ]
reference_data['D20']['G2']=[   372.057,
                                359.413,
                                334.775,
                                310.663,
                                287.100,
                                264.475,
                                242.475,
                                220.938,
                                200.438,
                                180.688,
                                161.600,
                                143.425,
                                125.925,
                                109.525,
                                94.225,
                                79.725,
                                66.113,
                                53.288,
                                42.238, 
                                ]
reference_data['D80']['G2']=[   366.386,
                                353.888,
                                329.413,
                                305.463,
                                281.800,
                                259.213,
                                237.313,
                                215.988,
                                195.725,
                                176.250,
                                157.538,
                                139.638,
                                122.550,
                                106.525,
                                91.425,
                                77.325,
                                64.125,
                                51.625,
                                40.838,
                                ]
reference_data['D90']['G2']=[   365.243,
                                352.788,
                                328.313,
                                304.375,
                                280.713,
                                258.138,
                                236.238,
                                214.975,
                                194.775,
                                175.338,
                                156.725,
                                138.925,
                                121.888,
                                105.925,
                                90.925,
                                76.825,
                                63.725,
                                51.325,
                                40.600,
                                ]

# create a ranger data object
class ranger():
    def __init__(self,rpath=None, RS=0, buildup=0, cal_data=None, ref_data=None):
        '''
        Initialise a Ranger data object with the following input parameters:
            rpath           - Directory MUST contain Ranger data (.bmp files and activescript.txt)
            RS              - Range shifter WET (mm)
            buildup         - Ranger buildup insert WET (mm)
            cal_data        - Parameters for analytical conversion of IDDs from pixels to mm (WET)
            ref_data        - Giraffe and TPS data used as baselines for QA
        '''
        self.rpath = rpath # path to ranger BeamworksStrata output files
        self.RS = RS # range shifter 
        self.buildup = buildup # buildup thickness
        
        # update calibration data
        self.calibration = {}
        if cal_data:
            for key in cal_data.keys():
                self.calibration[key]=cal_data[key]
        else:
            self.calibration=calibration
        
        # update reference data
        self.reference_data = reference_data
        if ref_data:
            for key in ref_data.keys():
                self.reference_data = ref_data[key]
    
    def load_data(self,rpath=None, RS=None, buildup=None):
        ''' load Ranger bmp images to obtain IDDs'''
        if not rpath:
            rpath = self.rpath
        else:
            self.rpath = rpath        
        if RS:
            self.RS = RS
        if buildup:
            self.buildup = buildup


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
        
        self.bmp_list = bmp_list
        self.idd() # generate IDDs
        self.idd_metrics() # analyse IDDs

        #check beam orientation in image stack
        self.check_orientation()
        if self.orientation != "BOTTOM":
            warnings.warn("Image orientation incorrect. Beam origin is "+self.orientation+", but should be from BOTTOM. IDD metrics will be unreliable.")

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
        plt.show()

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

    def plot_idd(self,i=-1, E=None,axis=-1):
        if axis == -1:
            x = np.array(range(self.raw_idd[0].shape[0]*-1,0))*-1
        else:
            x = np.array(range(self.raw_idd[0].shape[0]))
        if len(self.raw_idd) >= i >=0:
            rng = range(i,i+1)
            single_idd=True
        else:
            rng = range(0,len(self.raw_idd))
            single_idd=False
        for k in rng:
            idd = copy(self.raw_idd[k])
            plt.plot(x,idd)
        plt.xlabel('Depth (px)')
        plt.ylabel('Dose (norm)')
        if isinstance(E,list):
            if len(E)>1:
                plt.title(str(E[0])+'-'+str(E[-1])+' MeV')
                plt.legend([str(en)+str(" MeV") for en in E], borderaxespad=0, loc='center left', bbox_to_anchor=(1.04,0.5))
            else:
                plt.title(str(E[0])+' MeV')
        elif single_idd:
            plt.title(os.path.basename(self.bmp_list[i]))
        plt.show()
    
    def idd2mm(self, buildup=None, RS=None):
        ''' convert IDDs from pixels to WET mm'''
        # re-assign buildup and range shifter values if necessary
        if buildup:
            self.buildup = buildup
        if self.buildup in ['PMMA 11.05','PMMA 11.10','PTFE 13.27']:
            self.buildup = reference_data[self.buildup]
        if RS:
            self.RS = RS
        if self.RS in ['RS 5cm','RS 3cm','RS 2cm']:
            self.RS = reference_data[self.RS]
        # assign px to mm parameters
        px_pitch = self.calibration['pixel_pitch']
        scint_wer = self.calibration['scintillator_WER']
        wind_thickness = self.calibration['window_thickness']
        wind_wer = self.calibration['window_WER']
        px_offset = self.calibration['pixel_offset']
        # convert pixel metrics to mm
        window_WET = wind_thickness/wind_wer/px_pitch # PTFE window screwed onto ranger
        peak_offset = px_offset/scint_wer # some notional offset
        range_offset = window_WET+peak_offset+self.buildup+self.RS # total range offset inlucing WET of RS and PTFE insert
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

    def check_orientation(self):
        '''check orientation of image stack and automatically correct'''
        img_stack = copy(self.img)
        img = np.squeeze(np.sum(img_stack,0))
        img = img[150:-150,150:-150]
        img[img<img.max()*0.2]=0
        Vprofile = np.sum(img,axis=-1)
        Hprofile = np.sum(img,axis=0)
        edges = [Vprofile[-1],Vprofile[0],Hprofile[-1],Hprofile[0]]
        l=[i for i in range(len(edges)) if edges[i] != 0]
        idx = l[0]
        if len(l)>1 or len(l)==0:
            s = "AMBIGUOUS"
        elif idx==0:
            s = "BOTTOM"
        elif idx==1:
            s = "TOP"
        elif idx==2:
            s = "RIGHT"
        elif idx==3:
            s = "LEFT"
        self.orientation = s

    def calibrate(self,cal_dict=None):
        '''calibrate ranger parameters using an optimizer'''
        if cal_dict:
            calibration = cal_dict
        else:
            calibration = copy(self.calibration)
        # assign calibration D80s
        D80_cal = calibration['CalD80'] # calibration reference D80s in mm
        D80_px = calibration['RangerD80'] # Ranger measurement D80s in px
        D80_cal = np.array(D80_cal)
        D80_px = np.array(D80_px)
        # assign calibration parameters
        X=[ calibration['pixel_pitch'],
            calibration['scintillator_WER'],
            calibration['window_thickness'],
            calibration['window_WER'],
            calibration['pixel_offset'] ]
        # assign fixed calibration args
        A = [ D80_px,
              D80_cal,
              calibration['buildup'],
              calibration['RS'] ]
        
        # optimisation function
        def objective(x, a):
            # fixed objective function params
            d80_px=a[0] # ranger d80 (pixels)
            d80_cal=a[1] # calibration reference D80 (mm)
            buildup=a[2] # buildup WET (mm)
            rs=a[3] # range shifter WET (mm)

            # tunable objective function params
            px_pitch=x[0] # pixels per mm
            scint_wer=x[1] # scintllator WER
            wind_thickness=x[2] # window thickness (pixels)
            wind_wer=x[3] # window WER
            px_offset=x[4] # some notional pixel offset

            # objective function terms
            window_WET=wind_thickness/wind_wer/px_pitch # PTFE window screwed onto ranger
            peak_offset = px_offset/scint_wer # some notional offset
            range_offset = window_WET+peak_offset+buildup+rs # total range offset inlucing WET of RS and PTFE insert
            ranger_mm = (d80_px-wind_thickness)/scint_wer/px_pitch+range_offset # ranger D80 term (mm)
            
            # objective function (SSD)
            return np.sum((ranger_mm-d80_cal)**2)
        
        # optimizer
        result = minimize(objective,x0=X,args=A, method='L-BFGS-B')
        newcal = {
            'pixel_pitch':result['x'][0],
            'scintillator_WER':result['x'][1],
            'window_thickness':result['x'][2],
            'window_WER':result['x'][3],
            'pixel_offset':result['x'][4],
            'results': result,
            'CalD80': calibration['CalD80'],
            'RangerD80': calibration['RangerD80'],
        }
        return newcal
