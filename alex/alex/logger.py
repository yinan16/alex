# ----------------------------------------------------------------------
# Created: tis jul 27 19:01:12 2021 (+0200)
# Last-Updated:
# Filename: logger.py
# Author: Yinan Yu
# Description:
# ----------------------------------------------------------------------
import logging


class Logger(logging.StreamHandler):

    def __init__(self, stream=None):
        self.colors = {"eom": "\x1b[0m",
                       logging.CRITICAL: "\x1b[31m",
                       logging.ERROR: "\x1b[31m",
                       logging.INFO: "\x1b[37m",
                       logging.WARNING: "\x1b[33m",
                       logging.DEBUG: "\x1b[34m"}

        logging.StreamHandler.__init__(self, stream)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.setFormatter(formatter)

    def get_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("pydot").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)
        logger.addHandler(self)
        return logger


    def format(self, record):
        msg = logging.StreamHandler.format(self, record)
        clr = self.colors[record.levelno]
        return clr + msg + self.colors["eom"]


logger = Logger().get_logger()


if __name__=="__main__":

    # Example
    logger.warning("warning")
    logger.error("error")
    logger.debug("debug")
    logger.info("info")
