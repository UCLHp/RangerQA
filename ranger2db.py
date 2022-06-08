from matplotlib.pyplot import table
import  pandas as pd 
import pypyodbc


spreadsheet = (r'\\10.140.79.12\rtp-share$\protons\Work in Progress\AlexG\RangerQA\2022-04-28_154031\results_2022-04-28_154031.csv')
DF = pd.read_csv (spreadsheet)

DB_PATH = r'\\10.140.79.12\rtp-share$\protons\Work in Progress\AlexG\Access\AssetsDatabase_be.accdb'
DATABASE_TABLE = 'AbdulTest'
PASSWORD = None 

db_connection_string = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;'%(DB_PATH)    
conn = pypyodbc.connect(db_connection_string) 


def write_results_data(conn,DF):
    """Write results to table"""    
    
    cursor = conn.cursor()   
    sql = '''
        INSERT INTO "%s" (Rindex, ADate, [Reference Data], Energy, D20, D80, D90, RangerD20, RangerD80, RangerD90,\
        [D20 diff], [D80 diff], [D90 diff], [D20 PASSFLAG], [D80 PASSFLAG], [D90 PASSFLAG])
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        '''%(DATABASE_TABLE)  
         
    print("Attempting to write results to database...")

    try:
        for index, row in DF.iterrows():
            data = [row['Rindex'], row['ADate'] , row['Reference Data'] , row['Energy'] , row['D20'], row['D80'], row['D90'], row['RangerD20'], row['RangerD80'], \
            row['RangerD90'], row['D20 diff'], row['D80 diff'], row['D90 diff'], row['D20 PASSFLAG'], row['D80 PASSFLAG'], row['D90 PASSFLAG']] 
            cursor.execute(sql, data)

    except: 
            pypyodbc.IntegrityError: ('23000', '[23000] [Microsoft][ODBC Microsoft Access Driver] The changes you requested to the table were not successful because they would create duplicate values in the index, primary key, or relationship. Change the data in the field or fields that contain duplicate data, remove the index, or redefine the index to permit duplicate entries and try again.')
            print('One or all primary keys already exist in table. Data cannot be duplicated.')
        
    conn.commit()
    cursor.close()

    print("Finished. Check database table.")



write_results_data(conn,DF)