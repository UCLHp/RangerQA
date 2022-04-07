import datetime
import time
import glob
import os
import csv
from webbrowser import BackgroundBrowser
import pandas as pd
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ranger_data import ranger

# Dummy data
G = ['Gantry 1','Gantry 2','Gantry 3','Gantry 4']
Op = ['AGr','SC','NA']
RS = ['None', '5 cm', '3 cm', '2 cm']
BU = ['None', 'PTFE 13.27', 'PMMA 11.05', 'PMMA 11.10']

# Reference data - should be loaded from DB in future
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
                T = sg.T(row, key=h.replace(" ","")+str(i), justification='right', text_color='white', size=size)
                c_layout.append([T])
            tab_cols.append(sg.Column(c_layout))
        tabs.append(sg.Tab(k,[tab_cols]))
    return tabs

# Action Levels
delta = [-0.5,0.5] # mm

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
    def __init__(self):
        self.D20=None,  # ranger mm
        self.D80=None,  # ranger mm
        self.D90=None,  # ranger mm
        self.refD20=None,  # reference data mm
        self.refD80=None,
        self.refD90=None,
        self.tpsD20=None,  # TPS data mm
        self.tpsD80=None,
        self.tpsD90=None,
        self.drefD20=None,  # difference to ref data mm
        self.drefD80=None,
        self.drefD90=None,
        self.dtpsD20=None,  # difference to TPS data mm
        self.dtpsD80=None,
        self.dtpsD90=None,
        self.threshold=delta # PASS / FAIL threshold mm
        self.result=None    # PASS / FAIL flag
    
    def diff_calc(self):
        '''calculate differences between ranger and reference data'''
        # calculate differences
        diffs = [
            np.array(self.D20) - np.array(self.refD20),
            np.array(self.D80) - np.array(self.refD80),
            np.array(self.D90) - np.array(self.refD90),
            np.array(self.D20) - np.array(self.tpsD20),
            np.array(self.D80) - np.array(self.tpsD80),
            np.array(self.D90) - np.array(self.tpsD90),
        ]
        # calculate pass/fail
        passfails=[]
        for d in diffs:
            dpf = []
            for i in len(d):
                if self.threshold[0] <= i <= self.threshold[1]:
                    dpf.append('PASS')
                else:
                    dpf.append('FAIL')
            passfails.append(dpf)


# check input field values are valid
def field_check(values):
    # write a function that checks every user-modified field
    pass_fields=True # return false if any test fails
    msg=0 # return a different number for each test
    
    # example:
    if values['-Op1-'] not in Op:
        sg.popup('Operator 1 invalid, please select from dropdown list.')
        return False, 1

    return pass_fields, msg


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
        [sg.T('Folder', size=(6,1), justification='right'), sg.Input(key='-dirLo-', size=(18,1), justification='right'), sg.FolderBrowse(button_text='210-70 MeV', size=(12,1))],
    ]
    sess3_layout = [
        [sg.T('Range Shifter', size=(13,1), justification='right'), sg.DD(RS, size=(6,1), enable_events=True, key='-rsLo-')],
    ]
    sess4_layout = [
        [sg.T('Buildup', size=(13,1), justification='right'), sg.DD(BU, size=(12,1), enable_events=True, key='-buLo-')],
    ]
    sess5_layout = [
        [sg.T('PASS', background_color='green', key='-spfLo-', justification='center', size=(6,1))],
    ]

    sess6_layout = [
        [sg.T('Folder', size=(6,1), justification='right'), sg.Input(key='-dirHi-', size=(18,1), justification='right'), sg.FolderBrowse(button_text='245-220 MeV', size=(12,1))],
    ]
    sess7_layout = [
        [sg.T('Range Shifter', size=(13,1), justification='right'), sg.DD(RS, size=(6,1), enable_events=True, key='-rsHi-')],
    ]
    sess8_layout = [
        [sg.T('Buildup', size=(13,1), justification='right'), sg.DD(BU, size=(12,1), enable_events=True, key='-buHi-')],
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

    layout_tabs = [sg.Tab('Ranger', layout1)]
    layout_tabs.extend(ref_layout(ref_data))

    layout = [[sg.TabGroup([layout_tabs])]]
    
    return sg.Window('Ranger QA', layout, finalize=True)


### Generate GUI
window = build_window()
session_analysed = False

while True:
    event, values = window.read()
    ### reset analysed flag if there is just about any event
    if event not in ['-Submit-','-AnalyseS-','-Export-','-ML-',sg.WIN_CLOSED]:
        session_analysed=False
        window['-CSV_WRITE-'](disabled=True) # disable csv export button
        window['-Submit-'](disabled=True) # disable access export button       
    
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
        # collect and analyse results
        session_analysed=True#False    
        # update GUI
        if session_analysed:
            print("Session analysed")
            window['-CSV_WRITE-'](disabled=False) # enable Export button
            window['-Submit-'](disabled=False) # enable Export button
            window['ADate'](disabled=True) # freeze session ID
    
    if event == '-Export-': ### Export results to csv
        if session_analysed and values['-Export-'] != '':
            print("Session exported")
        elif values['-Export-'] == '':
            sg.popup('Directory Not Selected', 'Choose a valid directory')
        else:
            sg.popup('Analysis Required', 'Analyse the session before exporting to csv')
            
    if event == '-Clear-': ### Clear GUI fields and results
        session_analysed=False
        print("Session cleared.")
        #except the following:
        except_list = ['-CalB-', '-CSV_WRITE-','figCanvas'] # calendar button text
        for key in values:
            if key not in except_list:
                window[key]('')
        window['ADate'](disabled=False) # freeze session ID
        window['-AnalyseS-'](disabled=False)

    if event == sg.WIN_CLOSED or event == '-Cancel-': ### user closes window or clicks cancel
        print("Session Ended.")
        break

window.close()



