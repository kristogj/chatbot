import os
import re
import codecs
import logging
import csv


def log_lines(file, n=10):
    """
    Preview n lines from file
    :param file: str
    :param n: int
    :return:
    """
    logging.info("Printing some lines from {}:".format(file))
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        logging.info("\t {}".format(line))


def load_lines(filename, fields):
    """
    Splits each line of the file into a dictionary of fields.
    Load a line formatted file into a dictionary of line objects.
    :param filename: str
    :param fields: list[str]
    :return: dict{lineID: line_obj}
    """
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]
            lines[line_obj['lineID']] = line_obj
    return lines


def load_conversations(filename, lines, fields):
    """
    Group fields of lines from 'load_lines' into conversations based on *movie_conversations.txt*
    Load a conversation formatted file into a list of conversation objects.
    :param filename: str
    :param lines:
    :param fields: list[str]
    :return: list[dict]
    """
    conversations = []
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]
            # Convert string to list (conv_obj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile('L[0-9]+')
            line_ids = utterance_id_pattern.findall(conv_obj["utteranceIDs"])
            # Reassemble lines
            conv_obj["lines"] = []
            for line_id in line_ids:
                conv_obj["lines"].append(lines[line_id])
            conversations.append(conv_obj)
    return conversations


def extract_sentence_pairs(conversations):
    """
    Extract paris of sentences from conversations. Return question answer pairs where the target line
    is the line after the input line in a conversation.
    :param conversations: list[conv_obj]
    :return: list[list[str,str]]
    """
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i + 1]["text"].strip()

            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs


def create_formatted_file(config):
    """
    Create a formatted .csv file of the data.
    :return: None
    """
    datafile = config["datafile"]

    delimiter = "~"

    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize field ids
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    logging.info("Processing corpus")
    lines = load_lines(os.path.join(config["corpus"], "movie_lines.txt"), MOVIE_LINES_FIELDS)
    logging.info("Loading conversations...")
    conversations = load_conversations(os.path.join(config["corpus"], "movie_conversations.txt"), lines,
                                       MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    logging.info("Writing newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter, lineterminator='\n')
        for pair in extract_sentence_pairs(conversations):
            writer.writerow(pair)

    # Print some samples of lines from the new file
    log_lines(datafile)
