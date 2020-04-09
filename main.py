from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import os

# Custom
from data import create_formatted_file
from utils import init_logger, load_config, load_encoder_decoder, load_optimizers, get_device
from vocabulary import load_prepare_data
from training import train_iterations
from models import GreedySearchDecoder
from objective import evaluate_input

if __name__ == '__main__':
    init_logger()
    config_path = "config.yaml"

    # Load settings for this run
    config = load_config(config_path)
    config["device"] = get_device()
    config["corpus"] = os.path.join("data", config["corpus_name"])

    # Path to formatted data file
    config["datafile"] = os.path.join(config["corpus"], "formatted_movie_lines.txt")

    # Make formatted file of data if it does not exist
    if not os.path.isfile("./data/cornell movie-dialogs corpus/formatted_movie_lines.txt"):
        create_formatted_file(config)

    # Load/Assemble Vocabulary and question-answer pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = load_prepare_data(config["datafile"], config)

    config["directory"] = os.path.join(save_dir, config["model_name"], config["corpus_name"],
                                       '{}-{}_{}'.format(config["encoder_n_layers"], config["decoder_n_layers"],
                                                         config["hidden_size"]))
    config["load_filename"] = os.path.join(config["directory"], '{}_checkpoint.tar'.format(config["checkpoint_iter"]))

    checkpoint = None
    if os.path.isfile(config["load_filename"]):
        # If loading on same machine the model was trained on
        checkpoint = torch.load(config["load_filename"])
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

    # Load
    encoder, decoder = load_encoder_decoder(voc, checkpoint, config)

    # If you want to train the model, run the training loop
    encoder_optimizer, decoder_optimizer = load_optimizers(encoder, decoder, checkpoint, config)
    train_iterations(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, checkpoint, config)

    # Now we can chat with the model. Set dropout layers to eval mode.
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Start chat
    evaluate_input(searcher, voc, config)
