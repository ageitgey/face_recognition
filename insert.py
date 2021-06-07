import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="db_face"
)

mycursor = mydb.cursor()

sql = "INSERT INTO check_temp (p_id,temp,check_in) VALUES (%s,%s,%s)"
val = ("John","36","1/1/2564")

mycursor.execute(sql,val)

mydb.commit()

print(mycursor.rowcount, "was inserted.")