import dlib
#import cv2
import face_recognition
from PIL import Image, ImageDraw
from PIL import ImageTk, Image
import psycopg2
import tkinter
from tkinter import filedialog
from tkinter import Tk, Label, Button
#####################################

class FaceRecognition:

    def __init__(self, master):
        self.master = master
        master.title("Face Recognition")

        self.leftframe = tkinter.Frame(root)
        self.rightframe = tkinter.Frame(root, width=400)
        root.minsize(800,400)
        self.leftframe.pack(side=tkinter.LEFT)
        self.rightframe.pack(side=tkinter.LEFT)

        # get defualt image
        self.FirstTimeSetDefault()

        #set up form
        label_1 = tkinter.Label(self.rightframe, text="Identity:")
        label_1.grid(row=1,column=0)

        label_2 = tkinter.Label(self.rightframe, text="First Name")
        label_2.grid(row=2,column=0)

        label_3 = tkinter.Label(self.rightframe, text="Last Name")
        label_3.grid(row=2,column=1)

        self.label_4 = tkinter.Label(self.rightframe, text="_________")
        self.label_4.grid(row=3,column=0)

        self.label_5 = tkinter.Label(self.rightframe, text="_________")
        self.label_5.grid(row=3,column=1)

        button1 = tkinter.Button(self.rightframe, text="Submit Image", fg="red", command=self.SubmitImage)
        button1.grid(row=0,column=0)

    def FirstTimeSetDefault(self):
        image = Image.open("/Users/melatti/Documents/Work/FaceRec/images/batman.jpeg")
        image = image.resize((400, 400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)

        #set up canvas
        self.canvas = tkinter.Canvas(self.leftframe, width=400, height=400,  bg='black')
        self.canvas.pack()
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.canvas.create_image(203, 203, image=img, anchor=tkinter.CENTER)
        self.canvas.image = img


    def SubmitImage(self):
        # Get image file location that we want to check
        self.filename = filedialog.askopenfilename(initialdir = "/Users/melatti/Documents/Work/FaceRec/Obama",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

        # get face encoding for 1 face in image (must modify for more than 1 face in image)
        image = face_recognition.load_image_file(self.filename)
        face_encoding = face_recognition.face_encodings(image)[0] #only 1 face expected when entering 'mugshot'



        face_encoding_string = ""

        for distance in face_encoding:
            face_encoding_string+=str(distance)
            face_encoding_string+=str(",")

        face_encoding_string = face_encoding_string[:-1]
        face_encoding_string += str("")

        #query database and get identies (with encodings)
        # example: SELECT c FROM test ORDER BY c <-> cube(array[0.5,0.5,0.5]) LIMIT 1;
        conn = psycopg2.connect(host="localhost",database="postgres", user="postgres", password="password")
        cur = conn.cursor()
        tempstring = "SELECT first_name, last_name FROM wanted ORDER BY face_encoding <-> cube(array["+face_encoding_string+"]) LIMIT 1"
        cur.execute(tempstring)
        response = cur.fetchall()
        print(cur.fetchall())
        #conn.commit()
        conn.close()

        #update labels
        self.label_4.configure(text = response[0][0]) # first name
        self.label_5.configure(text = response[0][1]) # last name
        name = response[0][0] + " " +response[0][1]

        #edit photo / display photo selected
        image = Image.open(self.filename)
        width, height = image.size
        ratio = 0
        if width > height:
            ratio = 400/width
        else:
            ratio = 400/height

        image = face_recognition.load_image_file(self.filename)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            #draw name onto photo
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        del draw #clean up

        pil_image = pil_image.resize((int(width*ratio), int(height*ratio)), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(203, 203, image=img, anchor=tkinter.CENTER)
        self.canvas.image = img


root = Tk()
my_gui = FaceRecognition(root)
root.mainloop()
