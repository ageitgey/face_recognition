from PIL import Image, ImageDraw
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import Tk, Label, Button
import os, numpy, scipy, dlib, face_recognition, psycopg2, tkinter

dir_path = os.path.dirname(os.path.realpath(__file__))

THRESHOLD = 0.6 # threshold, declare that 2 faces are not a close enough match

class FaceRecognition:

    def __init__(self, master):
        self.master = master
        master.title("Face Recognition")

        # Set up tkinter app interface
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
        # load default image into app interface
        image = Image.open(dir_path+"/images/batman.jpeg")
        image = image.resize((400, 400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)

        #set up canvas
        self.canvas = tkinter.Canvas(self.leftframe, width=400, height=400,  bg='black')
        self.canvas.pack()
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.canvas.create_image(203, 203, image=img, anchor=tkinter.CENTER)
        self.canvas.image = img

    def SubmitImage(self):
        # Get image file location that we want to complete face recognition on
        self.filename = filedialog.askopenfilename(initialdir = dir_path+"/Demo",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

        #connect db
        conn = psycopg2.connect(host="localhost",database="postgres", user="postgres", password="password")
        cur = conn.cursor()

        # display photo selected in app with drawings on the photo
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

        #all responses stored in list
        response_list = []

        # loops through all detected faces and query db for closest match face encoding
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            # build face encoding query
            face_encoding_string = ""
            for float_distance in face_encoding:
                face_encoding_string += str(float_distance)
                face_encoding_string += str(",")
            face_encoding_string = face_encoding_string[:-1]
            # finished building face encoding query
            # now query the database search for closest match face encoding (spatial geometry database query)
            query = "SELECT first_name, last_name, face_encoding FROM wanted ORDER BY face_encoding <-> cube(array["+face_encoding_string+"]) LIMIT 1"
            cur.execute(query) #execute query
            response = cur.fetchone()
            response = [response[0], response[1],response[2]]

            # need to check a threshold here to see if its even close enough...
            # need to compare face_ending with response[2] (face_ending from db)
            returned_face_encoding = response[2] # array of returned face encodings
            returned_face_encoding = returned_face_encoding.replace("(","")
            returned_face_encoding = returned_face_encoding.replace(")","")
            returned_face_encoding = [float(x) for x in returned_face_encoding.split(",")]

            A = numpy.array(face_encoding)
            B = numpy.array(returned_face_encoding)
            distance = scipy.spatial.distance.euclidean(A, B)
            print(distance)

            if distance > THRESHOLD:
                response[0] = "Unknown"
                response[1] = "Unknown"

            name = response[0] + " " + response[1]
            response_list.append(response)

            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            #draw name onto photo
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        del draw #clean up
        conn.close() #close database connection

        # print all discovered names to app labels
        firstname = ""
        lastname = ""
        for response in response_list:
            firstname += response[0] + "\n"
            lastname += response[1] + "\n"
        self.label_4.configure(text = firstname) # first name
        self.label_5.configure(text = lastname) # last name

        # put photo with drawings on the canvas in the app view
        pil_image = pil_image.resize((int(width*ratio), int(height*ratio)), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(203, 203, image=img, anchor=tkinter.CENTER)
        self.canvas.image = img


root = Tk()
my_gui = FaceRecognition(root)
root.mainloop()
