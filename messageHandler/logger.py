import logging

level = logging.INFO
#### Configuring logger ####
# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(level)

# Create a console handler and set its level and formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(level)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'))

# Add the console handler to the logger
logger.addHandler(console_handler)

# Create a file handler and set its level and formatter
file_handler = logging.FileHandler('application.log', mode='w')  # specify the log file name
file_handler.setLevel(level)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'))

# Add the file handler to the logger
logger.addHandler(file_handler)


def log_message(log_level, message):
    # Log the message at the specified level
    if log_level == 'info':
        logger.info(message)
    elif log_level == 'debug':
        logger.debug(message)
    elif log_level == 'error':
        logger.error(message)
    elif log_level == 'warning':
        logger.warning(message)


def info(message):
    log_message('info', message)


def debug(message):
    log_message('debug', message)


def warning(message):
    log_message('warning', message)


def error(message):
    log_message('error', message)

def close_handler():
    file_handler.close()
