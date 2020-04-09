# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End of sentence token


class Vocabulary:
    """
    The Vocabulary class keeps a mapping from words to indexes, a reverse mapping of indexes to words,
    a count of each word and a total word count. The class also provides methods for adding a word to the
    vocabulary, adding all words in a sentence and trimming infrequently seen words.
    """

    def __init__(self):
        pass
