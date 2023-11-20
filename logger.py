import logging

# Configure the logging system
logger = logging.getLogger(__name__)

# Setting up the logging configuration
# 'logging.basicConfig' configures the logging system.
# 'level=logging.INFO' means it will capture all log messages at INFO level and above.
# The format specified will include the time of the log message and the actual message content.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')