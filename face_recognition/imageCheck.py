# Import necessary libraries
from _future_ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import sys
import itertools
from gtts import gTTS
from playsound import playsound

# Define a function to print the location of a face and speak it out loud
def print_result(filename, location):
    top, right, bottom, left = location
    # Format the location as a string with the format "filename,top,right,bottom,left"
    result = "{},{},{},{},{}".format(filename, top, right, bottom, left)
    # Print the location to the console
    print(result)
    # Speak the location out loud using Google Text-to-Speech and the playsound library
    speak(result)

# Define a function to speak text using Google Text-to-Speech and the playsound library
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3')
    playsound('output.mp3')
    os.remove('output.mp3')

# Define a function to test an image for faces
def test_image(image_to_check, model, upsample):
    unknown_image = face_recognition.load_image_file(image_to_check)
    face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=upsample, model=model)
    # For each face location, print the location and speak it out loud
    for face_location in face_locations:
        print_result(image_to_check, face_location)

# Define a function to get a list of all image files in a folder
def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

# Define a function to process a list of image files using multiple processes
def process_images_in_process_pool(images_to_check, number_of_cpus, model, upsample):
    # Set the number of processes to use based on the number of CPU cores available
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # Create a process pool using the multiprocessing library
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")
    pool = context.Pool(processes=processes)

    # Map the test_image function to each image file in the list using the process pool
    function_parameters = zip(
        images_to_check,
        itertools.repeat(model),
        itertools.repeat(upsample),
    )
    pool.starmap(test_image, function_parameters)

# Define the main function that will be run when the script is executed
@click.command()
@click.argument('image_to_check')
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel. -1 means "use all in system"')
@click.option('--model', default="hog", help='Which face detection model to use. Options are "hog" or "cnn".')
@click.option('--upsample', default=0, help='How many times to upsample the image looking for faces. Higher numbers find smaller faces.')
def main(image_to_check, cpus, model, upsample):
    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, model, upsample) for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), cpus, model, upsample)
    else:
        test_image(image_to_check, model, upsample)


if _name_ == "_main_":
    main()