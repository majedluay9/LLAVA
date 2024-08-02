import cv2
import os
from tqdm import tqdm

import re

def sorted_alph_numeric( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def create_video_from_frames(frame_folder, output_video, frame_rate=10):
    # List all files in the directory and sort them in ascending order
    images = [img for img in sorted_alph_numeric(os.listdir(frame_folder)) if img.endswith(".png")]
    # Determine the width and height from the first image
    frame_path = os.path.join(frame_folder, images[0])
    frame = cv2.imread(frame_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Loop through all images
    for image in tqdm(images):
        img_path = os.path.join(frame_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)  # Write out the frame

    out.release()  # Release the video writer


if __name__ == '__main__':
    path = r"D:\Frames\LLAVA\REAL2_3600S_60s_IN_BYTES\Ports_matrix"
    output = r"D:\Frames\LLAVA\REAL2_IN_BYTES_Ports_matrix.mp4"

    create_video_from_frames(path,output)