from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("biden.jpg")

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

import math

# Create a list of all found faces
face_images = []
for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    face_images.append(Image.fromarray(face_image))

if face_images:
    # Build a grid of the faces
    n = len(face_images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Use a fixed thumb size for the grid
    thumb_size = 150
    grid_image = Image.new('RGB', (cols * thumb_size, rows * thumb_size))

    for i, face in enumerate(face_images):
        face = face.resize((thumb_size, thumb_size))
        grid_image.paste(face, ((i % cols) * thumb_size, (i // cols) * thumb_size))

    # Show the final grid of faces
    grid_image.show()
