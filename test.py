from os import name
import mysql.connector
import datetime

name = "PIN"
d = datetime.datetime.now()
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="db_face"
)
mycursor = mydb.cursor()

sql = "INSERT INTO check_temp (p_id,temp,check_in) VALUES (%s,%s,%s)"
val = (name,"36",d)
mycursor.execute(sql,val)
mydb.commit()
print(mycursor.rowcount, "was inserted.")


sql1 = 'SELECT p_id FROM check_temp WHERE p_id = (%s) '
mycursor.execute(sql1,[name])
myresult = mycursor.fetchall()
print(myresult)
i =0

for x in myresult:
  print(x)
  if x[i] == name:
    print("Pin Success")

  else:
    print("Pin Fail")
  i+1