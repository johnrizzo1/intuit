"""
Module to completely disable logging in the entire application.
This module should be imported before any other modules.
"""
import logging
import sys
import os

# Create a NullHandler that does nothing with log records
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# Create a custom filter that blocks all log messages
class BlockAllFilter(logging.Filter):
    def filter(self, record):
        return False

# Completely disable all logging
null_handler = NullHandler()
block_filter = BlockAllFilter()

# Apply to root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)  # Set to highest level
root_logger.addHandler(null_handler)
root_logger.addFilter(block_filter)

# Remove all existing handlers from the root logger
for handler in list(root_logger.handlers):
    if handler != null_handler:
        root_logger.removeHandler(handler)

# Monkey patch the logging module to prevent other libraries from changing the configuration
original_basicConfig = logging.basicConfig
def patched_basicConfig(**kwargs):
    # Do nothing to prevent other libraries from changing the config
    pass
logging.basicConfig = patched_basicConfig

# Monkey patch the logging.getLogger function to always return a disabled logger
original_getLogger = logging.getLogger
def patched_getLogger(name=None):
    logger = original_getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.addHandler(null_handler)
    logger.addFilter(block_filter)
    logger.propagate = False
    return logger
logging.getLogger = patched_getLogger

# Monkey patch the logging.Logger.setLevel method to do nothing
original_setLevel = logging.Logger.setLevel
def patched_setLevel(self, level):
    # Always keep it at CRITICAL
    original_setLevel(self, logging.CRITICAL)
logging.Logger.setLevel = patched_setLevel

# Redirect stdout and stderr to null device when INTUIT_QUIET environment variable is set
if os.environ.get('INTUIT_QUIET') == '1':
    class NullWriter:
        def write(self, s):
            pass
        def flush(self):
            pass
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Replace with null writers
    sys.stdout = NullWriter()
    sys.stderr = NullWriter()
    
    # Function to restore original stdout and stderr
    def restore_output():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
else:
    def restore_output():
        pass

# Export the restore_output function
__all__ = ['restore_output']