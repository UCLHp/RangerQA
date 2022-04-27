from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, TableStyle, Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pandas as pd
import os

def table_maker(df):
    t = [['Energy MeV',
    'Ref D20',
    'Ref D80',
    'Ref D90',
    'Rng D20',
    'Rng D80',
    'Rng D90',
    'D20 diff',
    'D80 diff',
    'D90 diff']]
    t_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('ALIGN',(0, 1), (0, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 8)])
    for row, values, in df.iterrows():
        r_list = [  values.Energy[:-3],
                    '%.2f' % values.D20,
                    '%.2f' % values.D80,
                    '%.2f' % values.D90,
                    '%.2f' % values.RangerD20,
                    '%.2f' % values.RangerD80,
                    '%.2f' % values.RangerD90,
                    '%.2f' % values['D20 diff'],
                    '%.2f' % values['D80 diff'],
                    '%.2f' % values['D90 diff']]
        t.append(r_list)
        passflags = [values['D20 PASSFLAG'],values['D80 PASSFLAG'],values['D90 PASSFLAG']]
        for column, value in enumerate(passflags):
            if value == 'PASS':
                t_style.add('BACKGROUND', (-3+column,row+1), (-3+column,row+1), colors.green)
            elif value == 'WARN':
                t_style.add('BACKGROUND', (-3+column,row+1), (-3+column,row+1), colors.orange)
            elif value == 'FAIL':
                t_style.add('BACKGROUND', (-3+column,row+1), (-3+column,row+1), colors.red)
    table = Table(t)
    table.setStyle(t_style)
    return table

def report_maker(results_dict=None,session_df=None,dirname='.'):
    #results dataframes
    rkeys = list(results_dict.keys())
    df1 = results_dict[rkeys[0]]
    df2 = results_dict[rkeys[1]]

    #results tables        
    t1 = table_maker(df1)
    t2 = table_maker(df2)

    # image element
    h = 6
    w = h*0.4294
    im=Image(os.path.join(dirname,'IDD_plots.jpg'), h*inch, w*inch)

    # Operator 2 processing
    if isinstance(session_df.Operator2[0], float):
        op2 = ''
    else:
        op2 = session_df.Operator2[0]

    # doc elements
    fname = os.path.join(dirname,'RangerReport.pdf')
    doc = SimpleDocTemplate(fname, pagesize=A4,
                            rightMargin=72,leftMargin=72,
                            topMargin=40,bottomMargin=18)
    Story = []
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Indent', alignment=TA_JUSTIFY, leftIndent=20))
    styles.add(ParagraphStyle(name='Underline', alignment=TA_JUSTIFY, underlineWidth=1))

    title = '<font size="14"><u>'+session_df.Gantry[0]+' Ranger QA Results</u></font>'
    Story.append(Paragraph(title, styles["Justify"]))
    Story.append(Spacer(1, 10))
    dateline_1 = '<font size="10">Session timestamp: {}</font>'.format(session_df.ADate[0])
    Story.append(Paragraph(dateline_1, styles["Indent"]))
    operatorline1 = '<font size="10">Operator 1: {}</font>'.format(session_df.Operator1[0])
    Story.append(Paragraph(operatorline1, styles["Indent"]))
    operatorline2 = '<font size="10">Operator 2: {}</font>'.format(op2)
    Story.append(Paragraph(operatorline2, styles["Indent"]))
    Story.append(Spacer(1, 10))

    title = '<font size="12"><u>Ranger IDDs:</u></font>'
    Story.append(Paragraph(title, styles["Justify"]))
    Story.append(Spacer(1, 10))
    Story.append(im)
    Story.append(Spacer(1, 10))
    title = '<font size="12"><u>Summary of Ranger Measurements:</u></font>'
    Story.append(Paragraph(title, styles["Justify"]))
    Story.append(Spacer(1, 10))
    rsline = '<font size="10"><u>245 MeV - 220 MeV:</u></font>'
    Story.append(Paragraph(rsline, styles["Indent"]))
    rsline = '<font size="8">Range Shifter: {}</font>'.format(session_df.RSHi[0])
    Story.append(Paragraph(rsline, styles["Indent"]))
    rsline = '<font size="8">Buildup: {}</font>'.format(session_df.BUHi[0])
    Story.append(Paragraph(rsline, styles["Indent"]))
    Story.append(Spacer(1, 5))
    rsline = '<font size="10"><u>210 MeV - 70 MeV:</u></font>'
    Story.append(Paragraph(rsline, styles["Indent"]))
    rsline = '<font size="8">Range Shifter: {}</font>'.format(session_df.RSLo[0])
    Story.append(Paragraph(rsline, styles["Indent"]))
    rsline = '<font size="8">Buildup: {}</font>'.format(session_df.BULo[0])
    Story.append(Paragraph(rsline, styles["Indent"]))
    Story.append(Spacer(1, 5))
    gantryline = '<font size="10"><u>{}</u></font>'.format(rkeys[0]+" Reference Comparison (mm):")
    Story.append(Paragraph(gantryline, styles["Justify"]))
    Story.append(Spacer(1, 5))
    Story.append(t1)
    Story.append(Spacer(1, 10))
    gantryline = '<font size="10"><u>{}</u></font>'.format(rkeys[1]+" Reference Comparison(mm):")
    Story.append(Paragraph(gantryline, styles["Justify"]))
    Story.append(Spacer(1, 5))
    Story.append(t2)
    doc.build(Story)

    print("Saved report: "+fname)
