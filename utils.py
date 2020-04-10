import logging
import torch
import torch.nn as nn
from models import EncoderRNN, LuongAttentionDecoderRNN
import yaml
import torch.optim as optim


def init_logger():
    """
    Initialize logger settings
    :return: None
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("app.log", mode="w"),
            logging.StreamHandler()
        ])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_config(path):
    """
    Load the configuration from config.yaml
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def load_encoder_decoder(voc, checkpoint, configs):
    """
    Initialize encoder and decoder, and load from file if prev states exists
    :param voc: Vocabulary
    :param checkpoint: dict
    :param configs: dict
    :return: Encoder, LuongAttentionDecoderRNN
    """
    logging.info('Building encoder and decoder ...')

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, configs["hidden_size"])

    # Initialize encoder & decoder models
    encoder = EncoderRNN(configs["hidden_size"], embedding, configs["encoder_n_layers"], configs["dropout"])
    decoder = LuongAttentionDecoderRNN(embedding, voc.num_words, configs)

    if checkpoint:
        voc.__dict__ = checkpoint['voc_dict']
        embedding.load_state_dict(checkpoint['embedding'])
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])

    logging.info('Models built and ready to go!')
    return encoder.to(get_device()), decoder.to(get_device())


def load_optimizers(encoder, decoder, checkpoint, configs):
    """
    Initialize optimizer for encoder decoder, and load from file if prev states exists.
    :param encoder: Encoder
    :param decoder: LuongAttentionDecoderRNN
    :param checkpoint: dict
    :param configs: dict
    :return: Adam, Adam
    """
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    logging.info('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=configs["learning_rate"])
    decoder_optimizer = optim.Adam(decoder.parameters(),
                                   lr=configs["learning_rate"] * configs["decoder_learning_ratio"])

    if checkpoint:
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    def set_cuda(lst):
        for state in lst:
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # If you have cuda, configure cuda to call
    if torch.cuda.is_available():
        set_cuda(encoder_optimizer.state_dict().values())
        set_cuda(decoder_optimizer.state_dict().values())
    return encoder_optimizer, decoder_optimizer
