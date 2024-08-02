import os

import pandas as pd
import matplotlib

import Functions
from VideoGeneration import create_video_from_frames
import PARAMETERS

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.sparse as sp
import scipy.spatial
import numpy as np
import sklearn.decomposition
from datetime import datetime
import utils as utils
from GLOBAL_Parameters import *

# # Generate Videos
output_dir = r"D:\Frames\LIAM\Complete_REAL3_3600S_60S_NoScale"
output_dir_ip = r"D:\Frames\LIAM\Complete_REAL3_3600S_60S_NoScale\ip"

create_video_from_frames(output_dir_ip, f'{output_dir}_IPs.mp4', frame_rate=10)


# # Empty the folders after Generating the videos
# print('Now deleting all frames after generating the videos ... ')
# Functions.empty_folder(output_dir_ip)
# Functions.empty_folder(output_dir_port)
# Functions.empty_folder(output_dir_all)
#
# print('Done')