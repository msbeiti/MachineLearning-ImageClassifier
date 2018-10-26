import logging
import os

    # ToDo: adjust log path
def create_logger(name="ImageClassifier", type="console", level="DEBUG", filename="./logs/imageclassifier.log"):
    '''Log information to filename according to log level.

        Valid log levels are DEBUG, INFO, WARN, ERROR, and Critical '''

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create a file or console handler and set level
    try:
        assert type in ("console", "file")
    except AssertionError:
        print("ERROR: Log type can be either console or file")
        raise SystemExit()
    if type == "console":
        handler = logging.StreamHandler()
    else:
        if not os.path.isdir("./logs"):
            os.makedirs("./logs")
        handler = logging.FileHandler(filename)
    try:
        assert level in ("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL")
    except AssertionError:
        print("ERROR Log level should be one of the following: DEBUG, INFO, WARN, ERROR, CRITICAL")
        raise SystemExit()
    handler.setLevel(level)

    # create formatter
    formatter = logging.Formatter(fmt='[%(asctime)s] %(name)s.%(levelname)s.|%(funcName)s|: %(message)s',
                                  datefmt='%m/%d/%Y %I:%M:%S %p')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Starting logger '%s'." % logger.name)
    return logger
