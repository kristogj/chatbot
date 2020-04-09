from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import os
import logging

# Custom
from data import create_formatted_file
from utils import init_logger
from data import CORPUS, CORPUS_NAME
from vocabulary import load_prepare_data

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

if __name__ == '__main__':
    init_logger()

    # Path to formatted data file
    datafile = os.path.join(CORPUS, "formatted_movie_lines.txt")

    # Make formatted file of data if it does not exist
    if not os.path.isfile("./data/cornell movie-dialogs corpus/formatted_movie_lines.txt"):
        create_formatted_file()

    # Load/Assemble Vocabulary and question-answer pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = load_prepare_data(CORPUS, CORPUS_NAME, datafile, save_dir)

    # Print some paris to validate
    logging.info("Some QA-pairs:")
    for pair in pairs[:10]:
        logging.info("\t {}".format(pair))
