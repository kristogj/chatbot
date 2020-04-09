import logging
import torch

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
