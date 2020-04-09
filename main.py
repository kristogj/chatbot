from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import os

# Custom
from data import create_formatted_file
from utils import init_logger

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

if __name__ == '__main__':
    init_logger()

    # Make formatted file of data if it does not exist
    if not os.path.isfile("./data/cornell movie-dialogs corpus/formatted_movie_lines.txt"):
        create_formatted_file()
    