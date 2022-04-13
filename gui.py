import datetime
from genericpath import isfile
import time
import glob
import os
import csv
import warnings
from webbrowser import BackgroundBrowser
import pandas as pd
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ranger_data import ranger

# WARN/FAIL tresholds Levels
delta = [0.5,1.0] # abs(mm)
passflag_colors = ['green','orange','red']
passflag_labels = ['PASS','WARN','FAIL']

# Dummy data
G  = ['Gantry 1','Gantry 2','Gantry 3','Gantry 4']
Op = ['AGr','SC','NA']
RS = ['None', '5 cm', '3 cm', '2 cm']
BU = ['None', 'PTFE 13.27', 'PMMA 11.05', 'PMMA 11.10']

# Load reference data from csv - should be loaded from DB in future
def refdata():
    ref_data = {}
    ref_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), 'reference_data'))
    fpaths = glob.glob(os.path.join(ref_dir,'*.csv'))
    refnames = [os.path.basename(i)[:-4] for i in fpaths]
    for n,fp in enumerate(fpaths):
        data=pd.read_csv(fp)
        data['RangerD20']=['']*data.shape[0]
        data['RangerD80']=['']*data.shape[0]
        data['RangerD90']=['']*data.shape[0]
        data['D20 diff']=['']*data.shape[0]
        data['D80 diff']=['']*data.shape[0]
        data['D90 diff']=['']*data.shape[0]
        ref_data[refnames[n]] = data
    return ref_data

ref_data = refdata() # dict of reference dataframes

# convert reference data dict to pysimplegui tabbed layout
def ref_layout(ref_data, size=(8,1)):
    tabs = []
    for k in ref_data.keys():
        tab_cols = []
        df = ref_data[k]
        headers = list(df.columns)
        for h in headers:
            col = df[h]
            c_layout = [[sg.T(h,justification='right', text_color='white', size=size)]]
            for i, row in enumerate(col):
                if isinstance(row,(int,float)):
                    row='%.4f' % float(row)
                T = sg.T(row, key=h.replace(" ","")+str(i)+'G'+k[-1], justification='right', text_color='white', size=size)
                c_layout.append([T])
            tab_cols.append(sg.Column(c_layout))
        tabs.append(sg.Tab(k,[tab_cols],visible=False, key=k))
    return tabs


# Session and Results classes
class RangerSession():
    def __init__(self):
        self.sdate=None,    # session date
        self.gantry=None,   # gantry name
        self.op1=None,  # operator 1
        self.op2=None,  # operator 2
        self.dirLo=None,    # Beamworks output directory 210-70 MeV
        self.dirHi=None,    # Beamworks output directory 220-245 MeV
        self.RSlo=None,     # range shifter
        self.RShi=None,     # range shifter
        self.BUlo=None,     # buildup
        self.BUhi=None,     # buildup
        self.session=None   # PASS / FAIL flag
    
    def assign(self, values):
        '''assign field values to class vars'''

    def field_check(self):
        '''check fields for mistakes '''
    
    def analyse(self):
        '''call ranger class and return metrics in mm'''


class RangerResults():
    def __init__(self,results_dict=None, delta=None):
        self.delta = delta
        self.results_dict=results_dict
        self.threshold=delta # PASS / FAIL threshold mm
        self.result=None    # PASS / FAIL flag
    
    def diff_calc(self):
        '''calculate differences between ranger and reference data'''
        passflags = []
        dpoints = ['D20','D80','D90']
        for _, df in self.results_dict.items():
            for d in dpoints:
                #calculate deltas
                dref = np.array( df[d].tolist() )
                drng = np.array( df['Ranger'+d].tolist() )
                diffs = list(dref-drng)
                df[d+' diff'] = diffs
                passflags = [passflag_labels[0] if abs(i) <= self.delta[0] else passflag_labels[1] if abs(i) <= self.delta[1] else passflag_labels[2] for i in diffs]
                df[d+' PASSFLAG'] = passflags


# check ranger data directory contains required files
def dir_check(data_dir=None, HiLo=None):
    bmp_list = glob.glob(os.path.join(data_dir,'*.bmp'))
    act_txt = glob.glob(os.path.join(data_dir,'activescript.txt'))
    lvs_ini = glob.glob(os.path.join(data_dir,'*.ini'))
    if HiLo == 'lo':
        bmp_number = 15
    elif HiLo == 'hi':
        bmp_number = 4
    else:
        bmp_number=None    
    # check number of bmp files
    if len(bmp_list) != bmp_number:
            warnings.warn("Incorrect number of .bmp files in ranger data directory, expected number: "+str(bmp_number))
            sg.popup("Ranger Data: "+data_dir,"Incorrect number of .bmp files in ranger data directory, expected number: "+str(bmp_number))
            return False
    # check presence of activescript file
    elif len(act_txt) != 1:
            warnings.warn("Make sure activescript.txt exists in ranger data directory")
            sg.popup("Ranger Data: "+data_dir,"Make sure activescript.txt exists in ranger data directory")
            return False
    # check presence of winlvs file
    elif len(lvs_ini) != 1:
            warnings.warn("Make sure activescript.txt exists in ranger data directory")
            sg.popup("Ranger Data: "+data_dir,"Make sure activescript.txt exists in ranger data directory")
            return False
    return True


# check input field values are valid
def field_check(values):
    # write a function that checks every user-modified field
    pass_fields=True # return false if any test fails
    msg=0 # return a different number for each test
    
    # example for Operator 1:
    if values['-Op1-'] not in Op:
        sg.popup('Operator 1 invalid, please select from dropdown list.')
        return False, 1
    
    # Operator 2

    # Gantry

    # Date

    # Folders

    # Range Shifters

    # Buildups

    return pass_fields, msg

# export results to csv
def export_csv(dict={},dname='.'):
    csvtime = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_dir = dname+os.sep+csvtime
    os.makedirs(csv_dir, exist_ok=True)
    for t, r in dict.items():
        fname = csv_dir+os.sep+t
        r.columns = r.columns.str.strip()
        r.to_csv(fname,index=False)
        print("Saved: "+fname)

# graphing
_VARS = {'fig_agg': False, 'pltFig': False}

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def format_fig(xData,yData):
    plt.plot(xData,yData,'--k',linewidth=1)
    plt.title('Ranger IDDs', fontsize=10, fontweight='bold',color='w')
    plt.xlabel('Depth (mm)', fontsize=8, fontweight='bold',color='w')
    plt.ylabel('Dose (norm.)', fontsize=8, fontweight='bold',color='w')
    plt.xlim([0, 450])
    plt.ylim([0, 1])
    plt.xticks(fontsize=8)
    plt.tick_params(colors='w')
    plt.grid(visible=True)
    plt.tight_layout()








# build GUI
def build_window():
    # theme
    sg.theme('Topanga')

    # figure
    img_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pistachios.png'))
    plt_layout = [[sg.Image(img_file)]]  

    # session
    sess1_layout = [
        [sg.T('Date', justification='right', size=(10,1)), sg.Input(key='ADate', size=(18,1)),
         sg.T('Operator 1', justification='right', size=(10,1)), sg.DD(Op, size=(18,1), key='-Op1-'),
         sg.T('Gantry', justification='right', size=(10,1)), sg.DD(G, size=(18,1), enable_events=True, key='-G-'),
        ],
        [sg.T('', size=(10,1)), sg.CalendarButton('dd/mm/yyyy hh:mm:ss', font=('size',9), target='ADate', format='%d/%m/%Y %H:%M:%S', close_when_date_chosen=True, no_titlebar=False, key='-CalB-'),
         sg.T('Operator 2', justification='right', size=(10,1)), sg.DD(Op, size=(18,1), key='-Op2-'),
        ],
    ]

    sess2_layout = [
        [sg.T('Folder', size=(6,1), justification='right'), sg.Input(key='-dirLo-', size=(18,1), justification='right'), sg.FolderBrowse(button_text='210-70 MeV', size=(12,1), key='LoBrowse')],
    ]
    sess3_layout = [
        [sg.T('Range Shifter', size=(13,1), justification='right'), sg.DD(RS, default_value=RS[0], size=(6,1), enable_events=True, key='-rsLo-')],
    ]
    sess4_layout = [
        [sg.T('Buildup', size=(13,1), justification='right'), sg.DD(BU, default_value=BU[0], size=(12,1), enable_events=True, key='-buLo-')],
    ]
    sess5_layout = [
        [sg.T('PASS', background_color='green', key='-spfLo-', justification='center', size=(6,1))],
    ]

    sess6_layout = [
        [sg.T('Folder', size=(6,1), justification='right'), sg.Input(key='-dirHi-', size=(18,1), justification='right'), sg.FolderBrowse(button_text='245-220 MeV', size=(12,1), key='HiBrowse')],
    ]
    sess7_layout = [
        [sg.T('Range Shifter', size=(13,1), justification='right'), sg.DD(RS, default_value=RS[1], size=(6,1), enable_events=True, key='-rsHi-')],
    ]
    sess8_layout = [
        [sg.T('Buildup', size=(13,1), justification='right'), sg.DD(BU, default_value=BU[1], size=(12,1), enable_events=True, key='-buHi-')],
    ]
    sess9_layout =[
        [sg.T('PASS', background_color='green', key='-spfHi-', justification='center', size=(6,1))],
    ]

    session_frame = sg.Frame('Session',[[sg.Column(sess1_layout)]])
    dataLo_frame = sg.Frame('210 MeV - 70 MeV', [[sg.Column(sess2_layout), sg.Column(sess3_layout),sg.Column(sess4_layout),sg.Column(sess5_layout)]])
    dataHi_frame = sg.Frame('245 MeV - 220 MeV', [[sg.Column(sess6_layout), sg.Column(sess7_layout),sg.Column(sess8_layout),sg.Column(sess9_layout)]])

    # buttons
    button_layout = [
        #sg.FileBrowse('Load Gantry Ref', target='-GRef-'), sg.In(key='-GRef-', enable_events=True, visible=False),
        #sg.FileBrowse('Load TPS Ref', target='-TRef-'), sg.In(key='-TRef-', enable_events=True, visible=False),
        sg.B('Submit to Database', disabled=True, key='-Submit-'),
        sg.B('Analyse Session', key='-AnalyseS-'),
        sg.FolderBrowse('Export to CSV', key='-CSV_WRITE-', disabled=True, target='-Export-'), sg.In(key='-Export-', enable_events=True, visible=False),
        sg.B('Clear', key='-Clear-'),
        sg.B('End Session', key='-Cancel-'),
    ]

    # plot figure frame
    fig_frame = sg.Frame('Results', [[sg.Column([[sg.Image(img_file,size=(1000,500))]], justification='center')]])
    # combine layout elements
    layout1 = [
        [session_frame],
        [dataLo_frame],
        [dataHi_frame],
        [fig_frame],
        button_layout,
    ]

    #laylay = [[sg.T('bar',key='BAR')],[ sg.B('Submit to my will', key='-foobar-')]]
    layout_tabs = [sg.Tab('Ranger', layout1, key='RangerTab')]#+ref_layout(ref_data)
    layout_tabs.extend(ref_layout(ref_data))

    layout = [[sg.TabGroup([layout_tabs], enable_events=True, key='TabGroup')]]
    
    return sg.Window('Ranger QA', layout, finalize=True)


### Generate GUI
window = build_window()
window['TPS'].update(visible=True)
session_analysed = False
analysisFlag = 0
while True:
    event, values = window.read()
    ### reset analysed flag if there is just about any event
    if event not in ['-Submit-','-AnalyseS-','-Export-','-ML-','TabGroup','RangerTab','TPS','Gantry 4','Gantry 3','Gantry 2','Gantry 1',sg.WIN_CLOSED]:
        session_analysed=False
        window['-CSV_WRITE-'](disabled=True) # disable csv export button
        window['-Submit-'](disabled=True) # disable access export button    
    
    if event == '-G-' and values['-G-'] in G:
        for gantry in G:
            if gantry is not values['-G-']:
                window[gantry].update(visible=False)
            else:
                window[gantry].update(visible=True)

    ### Button events
    if event == '-Submit-': ### Submit data to database
        if session_analysed:
            checked, msg = field_check(values)
            if checked:    
                print('Data submitted to database.')
                session_analysed=True
                window['-Submit-'](disabled=True) # disable access export button
                window['-AnalyseS-'](disabled=True) # disable access export button
                window['ADate'](disabled=True) # freeze session ID
            else:
                session_analysed = False
                window['-Submit-'](disabled=False) # disable access export button
                print(msg)
        else:
            sg.popup('Analysis Required', 'Analyse the session before submitting to database')

    if event == '-AnalyseS-': ### Analyse results

        # select reference data for results analysis
        rg = values['-G-']
        if rg not in G:
            session_analysed=False
            sg.popup("Invalid Gantry","Select a valid gantry from the dropdown list.")
        else:
            session_analysed=True
            results_dict = ref_data
            if analysisFlag==0:
                for gantry in G:
                    if gantry != rg and gantry != 'TPS':
                        del results_dict[gantry]
                analysisFlag=1

        # check ranger data directories
        ld = dir_check(data_dir=values['-dirLo-'], HiLo='lo')
        hd = dir_check(data_dir=values['-dirHi-'], HiLo='hi')
        if ld != True or hd != True:
            session_analysed=False
        
        # load ranger data
        if session_analysed:
            try:
                rlo.__init__()
                rhi.__init__()
            except:
                rlo = ranger()
                rhi = ranger()
            try:
                rlo.load_data(rpath=values['-dirLo-'])
                rhi.load_data(rpath=values['-dirHi-'],RS='RS 5cm', buildup='PTFE 13.27')
                session_analysed = True
            except:
                session_analysed = False

        if session_analysed:
            try:
                # assign IDD metrics to reference dataframes
                results_dict[rg]['RangerD20'] = rhi.metrics_mm['D20'].tolist() + rlo.metrics_mm['D20'].tolist()
                results_dict[rg]['RangerD80'] = rhi.metrics_mm['D80'].tolist() + rlo.metrics_mm['D80'].tolist()
                results_dict[rg]['RangerD90'] = rhi.metrics_mm['D90'].tolist() + rlo.metrics_mm['D90'].tolist()
                results_dict['TPS']['RangerD20'] = rhi.metrics_mm['D20'].tolist() + rlo.metrics_mm['D20'].tolist()
                results_dict['TPS']['RangerD80'] = rhi.metrics_mm['D80'].tolist() + rlo.metrics_mm['D80'].tolist()
                results_dict['TPS']['RangerD90'] = rhi.metrics_mm['D90'].tolist() + rlo.metrics_mm['D90'].tolist()
                # analyse results
                results = RangerResults(results_dict=results_dict,delta=delta)
                results.diff_calc()
            except:
                session_analysed = False
                sg.popup("WARNING!","Could not perform IDD analysis, check data.")
            
        # update results Tab
        hiflag = 0
        loflag = 0
        if session_analysed:
            for t, r in results.results_dict.items():
                for i in range(r.shape[0]):
                    rangerkey = str(i)+'G'+t[-1]
                    diffkey = str(i)+'G'+t[-1]
                    window['RangerD20'+rangerkey]( '%.4f' % r['RangerD20'][i] )
                    window['RangerD80'+rangerkey]( '%.4f' % r['RangerD80'][i] )
                    window['RangerD90'+rangerkey]( '%.4f' % r['RangerD90'][i] )
                    window['D20diff'+diffkey]( '%.4f' % r['D20 diff'][i], background_color= passflag_colors[passflag_labels.index(r['D20 PASSFLAG'][i])])
                    window['D80diff'+diffkey]( '%.4f' % r['D80 diff'][i], background_color= passflag_colors[passflag_labels.index(r['D80 PASSFLAG'][i])])
                    window['D90diff'+diffkey]( '%.4f' % r['D90 diff'][i], background_color= passflag_colors[passflag_labels.index(r['D90 PASSFLAG'][i])])
                    if r['D20 PASSFLAG'][i] == 'FAIL' or r['D80 PASSFLAG'][i] == 'FAIL' or r['D90 PASSFLAG'][i] == 'FAIL':
                        if i < 4:
                            window['-spfHi-']('FAIL', background_color='red')
                            hiflag = 1
                        else:
                            window['-spfLo-']('FAIL', background_color='red')
                            loflag = 1
                    elif (r['D20 PASSFLAG'][i] == 'WARN' or r['D80 PASSFLAG'][i] == 'WARN' or r['D90 PASSFLAG'][i] == 'WARN'):
                        if i < 4 and hiflag==0:
                            window['-spfHi-']('WARN', background_color='orange')
                        elif loflag==0:
                            window['-spfLo-']('WARN', background_color='orange')
                    
        # update GUI
            print("Session analysed")
            window['-CSV_WRITE-'](disabled=False) # enable Export button
            window['-Submit-'](disabled=False) # enable Export button
            window['ADate'](disabled=True) # freeze session ID
    
    if event == '-Export-': ### Export results to csv
        if session_analysed and values['-Export-'] != '' and os.path.isdir(values['-Export-']):
            export_csv(results.results_dict,values['-Export-'])
            print("Session exported")
        elif values['-Export-'] == '':
            sg.popup('Directory Not Selected', 'Choose a valid directory')
        elif os.path.isdir(values['-Export-']) is False:
            sg.popup('Directory Not Selected', 'Choose a valid directory')
        else:
            sg.popup('Analysis Required', 'Analyse the session before exporting to csv')

    if event == '-Clear-': ### Clear GUI fields and results
        session_analysed=False
        print("Session cleared.")
        #except the following:
        except_list = ['-CalB-', '-CSV_WRITE-','figCanvas','LoBrowse','HiBrowse'] # calendar button text
        for key in values:
            if key not in except_list:
                window[key]('')
        window['ADate'](disabled=False) # freeze session ID
        window['-AnalyseS-'](disabled=False)

    if event == sg.WIN_CLOSED or event == '-Cancel-': ### user closes window or clicks cancel
        print("Session Ended.")
        break

window.close()




