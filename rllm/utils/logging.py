import logging


class DuplicateLoggingFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg in self.seen:
            return False
        self.seen.add(record.msg)
        return True
