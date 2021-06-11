from os import name
import mysql.connector
import datetime

name = "PIN"

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="db_face"
)
mycursor = mydb.cursor()
sql1 = "SELECT p_id FROM check_temp WHERE p_id = '%s' "
mycursor.executemany(sql1,name)
myresult = mycursor.fetchall()

for x in myresult:
  print("x = "+ x)