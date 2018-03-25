import dlib, face_recognition, os
from PIL import ImageTk, Image
import psycopg2
import tkinter
from tkinter import filedialog
from tkinter import Tk, Label, Button

dir_path = os.path.dirname(os.path.realpath(__file__))

class LoadDatabase:

    def __init__(self, master):
        self.master = master
        master.title("Load Face Database")

        self.topframe = tkinter.Frame(root)
        self.bottomframe = tkinter.Frame(root)
        self.topframe.pack()
        self.bottomframe.pack(side=tkinter.BOTTOM)

        # get defualt image
        self.FirstTimeSetDefault()

        #set up form
        tkinter.Label(self.bottomframe, text="First Name").grid(row=1,column=0)
        tkinter.Label(self.bottomframe, text="Last Name").grid(row=2,column=0)
        self.entry_1 = tkinter.Entry(self.bottomframe)
        self.entry_1.grid(row=1,column=1)
        self.entry_2 = tkinter.Entry(self.bottomframe)
        self.entry_2.grid(row=2,column=1)
        button1 = tkinter.Button(self.bottomframe, text="Select Image", fg="red", command=self.ChooseImage).grid(row=0,column=0)
        button2 = tkinter.Button(self.bottomframe, text="Submit Database", fg="red", command=self.SubmitDataBase).grid(row=3, column=0)

    def FirstTimeSetDefault(self):
        image = Image.open(dir_path+"/images/batman.jpeg")
        image = image.resize((400, 400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)

        #set up canvas
        self.canvas = tkinter.Canvas(self.topframe, width=400, height=400,  bg='black')
        self.canvas.pack()
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.canvas.create_image(203, 203, image=img, anchor=tkinter.CENTER)
        self.canvas.image = img

    def SetDefault(self):
        self.filename = ""
        image = Image.open(dir_path+"/images/batman.jpeg")
        width, height = image.size
        ratio = 0
        if width > height:
            ratio = 400/width
        else:
            ratio = 400/height
        image = image.resize((int(width*ratio), int(height*ratio)), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        self.canvas.create_image(203, 203, image=img, anchor=tkinter.CENTER)
        self.canvas.image = img
        self.entry_1.delete(0, tkinter.END)
        self.entry_2.delete(0, tkinter.END)

    def ChooseImage(self):
        self.filename = filedialog.askopenfilename(initialdir = dir_path+"/Demo/Database",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        image = Image.open(self.filename)
        width, height = image.size
        ratio = 0
        if width > height:
            ratio = 400/width
        else:
            ratio = 400/height
        image = image.resize((int(width*ratio), int(height*ratio)), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        self.canvas.create_image(203, 203, image=img, anchor=tkinter.CENTER)
        self.canvas.image = img

    def SubmitDataBase(self):
        image = face_recognition.load_image_file(self.filename)
        face_encoding = face_recognition.face_encodings(image)[0] #only 1 face expected when entering 'mugshot'
        face_encoding_string = "("

        for distance in face_encoding:
            face_encoding_string+=str(distance)
            face_encoding_string+=str(",")

        face_encoding_string = face_encoding_string[:-1]
        face_encoding_string += str(")")
        print(face_encoding_string) # demo what a face encoding data looks like in the output
        conn = psycopg2.connect(host="localhost",database="postgres", user="postgres", password="password")
        cur = conn.cursor()
        cur.execute("INSERT INTO wanted (first_name, last_name, face_encoding) VALUES ('"+self.entry_1.get()+"', '"+self.entry_2.get()+"', '"+face_encoding_string+"')");
        conn.commit()
        conn.close()
        self.SetDefault()

root = Tk()
my_gui = LoadDatabase(root)
root.mainloop()
