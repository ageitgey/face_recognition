from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("biden.jpg")

# Find all the faces in the image using a pre-trained convolutional neural network.
# This method is more accurate than the default HOG model, but it's slower
# unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
# this will use GPU acceleration and perform well.
# See also: find_faces_in_picture.py
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
face_images: list = []
max_width: int = 0
max_height: int = 0
print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    face_images.append(pil_image)
    # Update max dimensions
    max_width = max(max_width, pil_image.width)
    max_height = max(max_height, pil_image.height)
    # pil_image.show()


# Create a collage
if face_images:
    # Determine collage layout
    num_faces = len(face_images)
    cols = min(3, num_faces)  # Max 3 columns
    rows = (num_faces + cols - 1) // cols
    
    # Create blank canvas
    collage_width = cols * max_width
    collage_height = rows * max_height
    collage = Image.new('RGB', (collage_width, collage_height), color='white')
    
    # Paste faces onto collage
    for idx, face_img in enumerate(face_images):
        row = idx // cols
        col = idx % cols
        
        # Resize face to standard size if needed
        face_img_resized = face_img.resize((max_width, max_height), Image.Resampling.LANCZOS)
        
        # Calculate position
        x = col * max_width
        y = row * max_height
        
        collage.paste(face_img_resized, (x, y))
    
    # Show the collage
    collage.show()
    # Optionally save it
    # collage.save("detected_faces.jpg")