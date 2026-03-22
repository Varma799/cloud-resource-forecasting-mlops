import sys


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        self.error_message = self.get_detailed_error_message(error_message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown_file"
        line_number = exc_tb.tb_lineno if exc_tb else "unknown_line"
        return f"Error occurred in python script [{file_name}] at line [{line_number}]: {str(error_message)}"

    def __str__(self) -> str:
        return self.error_message