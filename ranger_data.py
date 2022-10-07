import os
import glob
import warnings
import re
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
            'pixel_pitch': 3.5331762559580806,
            'scintillator_WER': 0.9549915289642864,
            'window_WER': 1.5257097725668383,
            'window_thickness': 8.401448046401558,   # window thickness in pixels
            'pixel_offset': 3.5561382831730874,
            'buildup': 0,
            'RS': 0,
            'simple': [2.67944407, 0.29823315, 0.99913209], # simple polynomial fit terms
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
            'RangerD80':[   944.4699332,
                            866.2717149,
                            793.0258475,
                            721.1216981,
                            652.5912317,
                            585.3165138,
                            522.0070175,
                            462.5039216,
                            403.65,
                            349.0386431,
                            299.6991018,
                            251.6478632,
                            206.6309706,
                            165.5074041,
                            126.9716878],
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
    'plateau_depth': 25.0, # mm
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
    def __init__(self,rpath=None, RS=0, buildup=0, cal_data=None, ref_data=None, simple_cal=True):
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
        self.simple_cal = simple_cal
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
        
        # reset all other data
        self.img = None
        self.saturated_pixels = None
        self.bmp_list = None
        self.fps = None
        self.gain = None
        self.hratio = None
        self.metrics = None
        self.metrics_mm = None
        self.orientation = None
        self.raw_idd = None
        self.roi_left = None
        self.roi_top = None
        self.roi_width = None
        self.roi_bottom = None
        self.shutter = None
        self.saturated_pixels = None
        self.x_centre = None
        self.y_centre = None


    # optimisation functions
    @staticmethod
    def objective_simple(x, a=None, obj=True):
        # tunable params
        if isinstance(x,list):
            x = np.array(x)
        # fixed params
        if isinstance(a[0],list):
            a[0] = np.array(a[0])
        # objective function terms (SSD)
        px2mm = (x[0]+x[1]*a[0]**x[2])+a[1]+a[2]
        if obj:
            ref_mm = a[3]
            ssd = np.sum((px2mm-ref_mm)**2)
            return ssd
        else:
            return px2mm
    
    @staticmethod
    def objective(x, a=None, obj=True):
        if isinstance(a[0],list):
            a[0] = np.array(a[0])
        # tunable objective function params
        px_pitch=x[0] # pixels per mm
        scint_wer=x[1] # scintllator WER
        wind_thickness=x[2] # window thickness (pixels)
        wind_wer=x[3] # window WER
        px_offset=x[4] # some notional pixel offset

        # fixed objective function params
        px=a[0] # ranger values (pixels)
        buildup=a[1] # buildup WET (mm)
        rs=a[2] # range shifter WET (mm)

        # objective function terms
        window_WET=wind_thickness/wind_wer/px_pitch # PTFE window screwed onto ranger
        peak_offset = px_offset/scint_wer # some notional offset
        range_offset = window_WET+peak_offset+buildup+rs # total range offset inlucing WET of RS and PTFE insert
        ranger_mm = (px-wind_thickness)/scint_wer/px_pitch+range_offset # ranger term (mm)
        
        if obj:
            cal=a[3] # calibration reference term (mm)
            # objective function (SSD)
            return np.sum((ranger_mm-cal)**2)
        else:
            return ranger_mm
    
    
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
        bmp_list.sort()
        img0 = Image.open(bmp_list[0])
        img0 = np.array(img0)
        self.img = np.empty((len(bmp_list),img0.shape[0],img0.shape[1]))
        self.img[0,:] = img0
        pbar = tqdm(bmp_list)
        for i, bmp in enumerate(pbar):
            pbar.set_description('Loading '+os.path.basename(bmp))
            self.img[i,:] = np.array(Image.open(bmp))
        saturation_mask = self.img>=255
        self.saturated_pixels = np.count_nonzero(saturation_mask)
        
        #check beam orientation in image stack
        self.check_orientation()
        if self.orientation == "TOP":
            self.img = np.flip(self.img,1)
        elif self.orientation != "BOTTOM":
            warnings.warn("Image orientation incorrect. Beam origin is "+self.orientation+", but should be from BOTTOM. IDD metrics may be unreliable.")
        
        with open(os.path.join(rpath,'activescript.txt')) as txt_info:
            lines = txt_info.readlines()
            c = 0
            for l in lines:
                if 'CameraHRatio=' in l:
                    self.hratio = float(re.sub('[^0-9.]','', l))
                if 'CameraVRatio=' in l:
                    self.vratio = float(re.sub('[^0-9.]','', l))
                if 'ROI_top=' in l:
                    self.roi_top = float(re.sub('[^0-9]','', l))
                    c+=1
                if 'ROI_left=' in l:
                    self.roi_left = float(re.sub('[^0-9]','', l))
                    c+=1
                if 'ROI_width=' in l:
                    self.roi_width = float(re.sub('[^0-9]','', l))
                    c+=1
                if 'ROI_bottom=' in l:
                    self.roi_bottom = float(re.sub('[^0-9]','', l))
                    c+=1
                if 'AppXCenter=' in l:
                    self.x_centre = float(re.sub('[^0-9.]','', l))
                    c+=1
                if 'AppYCenter=' in l:
                    self.y_centre = float(re.sub('[^0-9.]','', l))
                    c+=1
                if c == 6:
                    break
        
        with open(os.path.join(rpath,'winlvsapp.ini')) as ini_info:
            lines = ini_info.readlines()
            c = 0
            for l in lines:
                if 'VideoFrameRate=' in l:
                    self.fps = float(re.sub('[^0-9.]','', l))
                    c+=1
                if 'VideoShutter=' in l:
                    self.shutter = float(re.sub('[^0-9.]','', l))
                    c+=1
                if 'VideoGain=' in l:
                    self.gain = float(re.sub('[^0-9.]','', l))
                    c+=1
                if c == 3:
                    break
        
        self.bmp_list = bmp_list
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

    def plot_idd(self,i=-1, E=None, axis=-1, units='px'):

        if len(self.raw_idd) >= i >=0:
            rng = range(i,i+1)
            single_idd=True
        else:
            rng = range(0,len(self.raw_idd))
            single_idd=False
        if units == 'mm':
            for x,y in self.idd_mm:
                plt.plot(x,y)
            plt.xlabel('Depth (mm)')
        else:
            if axis == -1:
                x = np.array(range(self.raw_idd[0].shape[0]*-1,0))*-1
            else:
                x = np.array(range(self.raw_idd[0].shape[0]))
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
    
    def idd2mm(self, buildup=None, RS=None, simple=None):
        ''' convert IDDs from pixels to WET mm'''
        # re-assign buildup, range shifter and calibration method values if necessary
        if buildup:
            self.buildup = buildup
        if self.buildup in ['PMMA 11.05','PMMA 11.10','PTFE 13.27']:
            self.buildup = reference_data[self.buildup]
        if RS:
            self.RS = RS
        if self.RS in ['RS 5cm','RS 3cm','RS 2cm']:
            self.RS = reference_data[self.RS]
        if isinstance(simple,bool):
            self.simple_cal=simple
            
        # assign calibration parameters
        if self.simple_cal:
            X=self.calibration['simple']
            convert2mm = self.objective_simple
        else:
            X=[ self.calibration['pixel_pitch'],
                self.calibration['scintillator_WER'],
                self.calibration['window_thickness'],
                self.calibration['window_WER'],
                self.calibration['pixel_offset'] ]
            convert2mm = self.objective

        # convert metrics to mm using obj function
        P100_WET = convert2mm(X,[self.metrics['P100'], self.buildup,self.RS],obj=False)
        P90_WET  = convert2mm(X,[self.metrics['P90'], self.buildup,self.RS],obj=False)
        P80_WET  = convert2mm(X,[self.metrics['P80'], self.buildup,self.RS],obj=False)
        D80_WET  = convert2mm(X,[self.metrics['D80'], self.buildup,self.RS],obj=False)
        D90_WET  = convert2mm(X,[self.metrics['D90'], self.buildup,self.RS],obj=False)
        D20_WET  = convert2mm(X,[self.metrics['D20'], self.buildup,self.RS],obj=False)
        self.metrics_mm = {'P100':P100_WET, 'P80':P80_WET, 'P90':P90_WET,
                            'D90':D90_WET, 'D80':D80_WET, 'D20':D20_WET}
        
        # convert IDDs to mm using obj function
        self.idd_mm = []
        for y in self.raw_idd:
            idx, _ = find_peaks(y, width=y.max()*0.1, prominence=y.max()*0.075, height=y.max()*0.2)
            x0 = idx.max()
            idd = y[:x0]
            idd = np.flip(idd)
            x = np.array(range(idd.shape[0]))
            x_mm=convert2mm(X, [x,  self.buildup,self.RS], obj=False)
            self.idd_mm.append([x_mm,idd])

    def idd_metrics(self, plot=False):
        ''' extract metrics from IDDs '''
        def _distal_prox(y,x100,x0):
            d90_flag = True
            d80_flag = True
            d20_flag = True
            for i in range(x100,0,-1):
                d=0.9
                if y[i]<y[x100]*d and d90_flag:
                    xp=[y[i],y[i+1]]
                    fp=[i,i+1]
                    d90_flag = False
                    d90 = np.interp(d,xp=xp,fp=fp)

                d=0.8
                if y[i]<y[x100]*d and d80_flag:
                    xp=[y[i],y[i+1]]
                    fp=[i,i+1]
                    d80_flag = False
                    d80 = np.interp(d,xp=xp,fp=fp)

                d=0.2
                if y[i]<y[x100]*d and d20_flag:
                    xp=[y[i],y[i+1]]
                    fp=[i,i+1]
                    d20_flag = False
                    d20 = np.interp(d,xp=xp,fp=fp)

            # find Proximal metrics
            p90_flag = True
            p80_flag = True
            p20_flag = True
            for i in range(x100,x0,1):
                d=0.9
                if y[i]<y[x100]*d and p90_flag:
                    xp=[y[i],y[i-1]]
                    fp=[i,i-1]
                    p90_flag = False
                    p90 = np.interp(d,xp=xp,fp=fp)

                d=0.8
                if y[i]<y[x100]*d and p80_flag:
                    xp=[y[i],y[i-1]]
                    fp=[i,i-1]
                    p80_flag = False
                    p80 = np.interp(d,xp=xp,fp=fp)

            x = [p80,p90,d90,d80,d20]
            return x

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
            xraw = _distal_prox(y,x100,x0)
            #Bragg peak (P100)
            metrics['P100'].append(x0 - x100)
            # P80
            metrics['P80'].append(x0-xraw[0])
            # P90
            metrics['P90'].append(x0-xraw[1])
            # D90
            metrics['D90'].append(x0-xraw[2])
            # D80
            metrics['D80'].append(x0-xraw[3])
            # D20
            metrics['D20'].append(x0-xraw[4])
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

    def calibrate(self,cal_dict=None, simple=False):
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
        if simple:
            X=calibration['simple']
        else:
            X=[ calibration['pixel_pitch'],
                calibration['scintillator_WER'],
                calibration['window_thickness'],
                calibration['window_WER'],
                calibration['pixel_offset'] ]
        
        # fixed calibration args
        A = [ D80_px,
              calibration['buildup'],
              calibration['RS'],
              D80_cal ]
        
        # optimizer
        if simple:
            objective_fn = self.objective_simple
        else:
            objective_fn = self.objective

        result = minimize(objective_fn,x0=X,args=A, method='L-BFGS-B')
        if simple:
            newcal = {
                'simple':result['x'],
                'results':result,
                'CalD80':calibration['CalD80'],
                'RangerD80':calibration['RangerD80'],
            }
        else:
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
