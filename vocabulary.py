from collections import defaultdict
import logging
import unicodedata
import re

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End of sentence token

MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3  # Minimum word count threshold for trimming


class Vocabulary:
    """
    The Vocabulary class keeps a mapping from words to indexes, a reverse mapping of indexes to words,
    a count of each word and a total word count. The class also provides methods for adding a word to the
    vocabulary, adding all words in a sentence and trimming infrequently seen words.
    """

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word_to_index = {}
        self.word_to_count = defaultdict(lambda: 0)
        self.index_to_word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Initialised with PAD, SOS, EOS

    def add_sentence(self, sentence):
        """
        Add all words in sentence to the vocabulary
        :param sentence: str
        :return: None
        """
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        """
        Add word to the vocabulary - word to index, index to word and increase count
        :param word:
        :return:
        """
        if word not in self.word_to_index.keys():
            self.word_to_index[word] = self.num_words
            self.index_to_word[self.num_words] = word
            self.num_words += 1
        self.word_to_count[word] += 1

    def trim(self, min_count):
        """
        Remove words which has a word count less than min_count
        :param min_count: int
        :return: None
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for word, count in self.word_to_count.items():
            if count >= min_count:
                keep_words.append(word)

        percentage = len(keep_words) / len(self.word_to_index)
        logging.info("Keep words: {} / {} = {:.4f}".format(len(keep_words), len(self.word_to_index), percentage))

        # Reinitialize dictionaries
        self.word_to_index = {}
        self.word_to_count = defaultdict(lambda: 0)
        self.index_to_word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.add_word(word)

    def trim_rare_words(self, pairs):
        """
        Trim words used under the MIN_COUNT from the Vocabulary
        :param pairs: list[list[str]]
        :return:
        """
        self.trim(MIN_COUNT)

        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence, output_sentence = pair[0], pair[1]
            keep_input, keep_output = True, True

            # Check input sentence
            for word in input_sentence.split(" "):
                if word not in self.word_to_index:
                    keep_input = False
                    break

            # Check output sentence
            for word in output_sentence.split(" "):
                if word not in self.word_to_index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        percentage = len(keep_pairs) / len(pairs)
        logging.info("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), percentage))
        return keep_pairs


def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII
    :param s: str
    :return:
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(sent):
    """
    Lowercase, trim and remove non-letter characters
    :param sent: str
    :return:
    """
    sent = unicode_to_ascii(sent.lower().strip())
    sent = re.sub(r"([.!?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z.!?]+", r" ", sent)
    sent = re.sub(r"\s+", r" ", sent).strip()
    return sent


def read_vocabularies(datafile, corpus_name):
    """
    Read question-answer pairs and return a Vocabulary object.
    :param datafile: str
    :param corpus_name: str
    :return:
    """
    logging.info("Reading lines...")

    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split("\n")

    # Split every line into pairs and normalize
    pairs = [[normalize_string(sent) for sent in line.split("~")] for line in lines]
    voc = Vocabulary(corpus_name)
    return voc, pairs


def filter_pair(pair):
    """
    Return True iff both sentences in pair are under the MAX_LENGTH threshold
    :param pair:
    :return:
    """
    # Input sequences need to preserve the last word for EOS token
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    """
    Filter pairs using filter_pair condition
    :param pairs: list[list[str]]
    :return: list[list[str]]
    """
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus, corpus_name, datafile, save_dir):
    """
    Using the functions defined above, return a populated Vocabulary object and pairs list
    :param corpus_name:
    :param datafile:
    :return: Vocabulary, list[list[str]]
    """
    logging.info("Start preparing training data ...")
    voc, pairs = read_vocabularies(datafile, corpus_name)

    logging.info("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)

    logging.info("Trimmed to {!s} sentence pairs".format(len(pairs)))
    logging.info("Counting words...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    logging.info("Counted words: {}".format(voc.num_words))

    logging.info("Trimming rare words")
    pairs = voc.trim_rare_words(pairs)

    return voc, pairs
