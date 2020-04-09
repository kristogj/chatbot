import logging


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
