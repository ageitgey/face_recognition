import face_recognition

# Load a sample picture and learn how to recognize it.
obama_small_image = face_recognition.load_image_file("obama_small.jpg")
obama_face_encoding_small = face_recognition.face_encodings(obama_small_image)[0]

# Load a sample picture and learn how to recognize it.
obama2_image = face_recognition.load_image_file("obama2.jpg")
obama2_face_encoding = face_recognition.face_encodings(obama2_image)[0]

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

encodings = [obama_face_encoding_small,
			obama2_face_encoding,
			obama_face_encoding,
			biden_face_encoding
			]


obama_image2 = face_recognition.load_image_file("obama_small.jpg")
obama_face_encoding2 = face_recognition.face_encodings(obama_image2)[0]



result = face_recognition.face_distance(encodings,obama_face_encoding2)
for r in result:
	if r<0.4 :
 		print("the same people, similarty : {}".format(1-r))
 	else:
 		print("not the same people, similarty : {}".format(1-r))
