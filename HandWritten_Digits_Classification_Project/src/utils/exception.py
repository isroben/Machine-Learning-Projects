import sys
from src.utils.logger import get_logger

logger = get_logger(__name__)

def error_message_detail(error, error_detail: sys):
    exc_type, exc_obj, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error: {str(error)}."
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return f"Error occured in python script [{file_name}] at line [{line_number}] : {str(error)}."


class CustomException(Exception):
    def _init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

        logger.error(self.error_message)

    def __str__(self):
        return self.error_message