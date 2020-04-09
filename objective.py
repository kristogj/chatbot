import torch
from utils import get_device
from vocabulary import indexes_from_sentence, normalize_string
import logging


def mask_nll_loss(inp, target, mask):
    """
    This loss function calculates the average negative log likelihood of the elements that
    correspond to a 1 in the mask tensor.
    :param inp:
    :param target:
    :param mask:
    :return:
    """
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(get_device())
    return loss, n_total.item()


def evaluate(searcher, voc, sentence, config):
    """
    Now that we have our decoding method defined, we can write functions for evaluating a string input sentence.
    The evaluate function manages the low-level process of handling the input sentence.
    :param searcher: GreedySearchDecoder
    :param voc: Vocabulary
    :param sentence: str
    :return: list[str]
    """
    # Format input sentence as a batch. Words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]  # batch_size == 1

    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    # Use appropriate device
    input_batch = input_batch.to(get_device())
    lengths = lengths.to(get_device())

    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, config["max_length"])

    # Indexes -> words
    decoded_words = [voc.index_to_word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(searcher, voc, config):
    while True:
        try:
            # Get input sentence
            input_sentence = input('> ')

            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break

            # Normalize
            input_sentence = normalize_string(input_sentence)

            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence, config)

            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            logging.error("Error: Encountered unknown word.")
