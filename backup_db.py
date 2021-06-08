import mysql.connector
import time
import datetime
import pipes
from pymysql import* 
import xlwt 
import pandas.io.sql as sql

from pymysql import*
import xlwt
import pandas.io.sql as sql
d = datetime.date.today()
timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)
# connect the mysql with the python
con=connect(user="root",password="",host="localhost",database="db_face")
# read the data
df=sql.read_sql('select * from check_temp',con)
# print the data
print(df)
#print(d)
# export the data into the excel sheet
backup_path = 'backup/'
df.to_excel(backup_path+timestr +'.xls')