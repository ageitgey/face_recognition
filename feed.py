import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="db_face"
)

mycursor = mydb.cursor()
sql1 = "SELECT * FROM check_temp"
mycursor.execute(sql1)
myresult = mycursor.fetchall()

for x in myresult:
    print(x)