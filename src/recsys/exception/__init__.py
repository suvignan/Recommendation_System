import sys


class RecsysException(Exception):
    """Wraps an error with the file and line where it really happened. Raise inside an except block."""

    def __init__(self, error: Exception):
        _, _, tb = sys.exc_info()
        if tb is None:                      # raised outside an except block: no location
            location = "unknown location"
        else:
            while tb.tb_next:               # walk to the deepest frame = the real origin
                tb = tb.tb_next
            location = f"{tb.tb_frame.f_code.co_filename}, line {tb.tb_lineno}"
        self.message = f"{type(error).__name__} in {location}: {error}"
        super().__init__(self.message)
