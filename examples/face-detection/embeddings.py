import sys

#sys.path.append('/home/shivam/Downloads/lib/python3.6/site-packages')
sys.path.append('/home/shivam/Downloads/face-recognition/lib/python3.6/site-packages')
import cv2 
import face_recognition
import pickle
name=input("enter name")
ref_id=input("enter id")

try:
	f=open("ref_name.pkl","rb")

	ref_dictt=pickle.load(f)
	f.close()
except:
	ref_dictt={}
ref_dictt[ref_id]=name


f=open("ref_name.pkl","wb")
pickle.dump(ref_dictt,f)
f.close()

try:
	f=open("ref_embed.pkl","rb")

	embed_dictt=pickle.load(f)
	f.close()
except:
	embed_dictt={}





for i in range(5):
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	while True:
	     
		check, frame = webcam.read()
		# print(check) #prints true as long as the webcam is running
		# print(frame) #prints matrix values of each framecd 
		cv2.imshow("Capturing", frame)
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		rgb_small_frame = small_frame[:, :, ::-1]
		
		key = cv2.waitKey(1)

		if key == ord('s') : 
			face_locations = face_recognition.face_locations(rgb_small_frame)
			if face_locations != []:

				# filename="photo.jpg"
				# cv2.imwrite(filename=filename, img=frame)
				# image = face_recognition.load_image_file(filename)
				# image = Image.fromarray(frame)
				# image = image.convert('RGB')
				face_encoding = face_recognition.face_encodings(frame)[0]
				if ref_id in embed_dictt:
					embed_dictt[ref_id]+=[face_encoding]
				else:
					embed_dictt[ref_id]=[face_encoding]
				webcam.release()
				# img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
				# img_new = cv2.imshow("Captured Image", img_new)
				cv2.waitKey(1)
				cv2.destroyAllWindows()     
				break
		elif key == ord('q'):
			print("Turning off camera.")
			webcam.release()
			print("Camera off.")
			print("Program ended.")
			cv2.destroyAllWindows()
			break
f=open("ref_embed.pkl","wb")
pickle.dump(embed_dictt,f)
f.close()