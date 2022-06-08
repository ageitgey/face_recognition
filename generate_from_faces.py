import os
import pyperclip
import time

Sleep = 1

Files = os.listdir("faces/ucitele/")
Names = []
Out = ""

for File in Files:
    Path = "faces/ucitele/" + File
    # print(Path)
    Name = File[:-1]
    Name = Name[:-1]
    Name = Name[:-1]
    Name = Name[:-1]
    Names.append(Name)

    Full = '{} = load.img("{}")'.format(Name, Path)

    Out = Out + "\n" + Full
pyperclip.copy(Out)
time.sleep(Sleep)
print(Out)
print()
print()



known_fn = "known_face_names = {}".format(Names)

known_fe = "known_face_encodings = {}".format(Names)
known_fe = known_fe.replace("'", "")

pyperclip.copy(known_fe)
time.sleep(Sleep)
print(known_fe)
print()
print()
pyperclip.copy(known_fn)
time.sleep(Sleep)
print(known_fn)


