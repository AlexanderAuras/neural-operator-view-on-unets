import logging
import re

from typing_extensions import override


class IndentingFormatter(logging.Formatter):
    @override
    def format(self, record: logging.LogRecord) -> str:
        # Find auto-indent depth
        if self._fmt is not None:
            if isinstance(self._style, logging.PercentStyle):  # pyright: ignore [reportUnnecessaryIsInstance]
                match = re.search(r"%\(message\)s", self._fmt)
            elif isinstance(self._style, logging.StringTemplateStyle):  # pyright: ignore [reportUnnecessaryIsInstance]
                match = re.search(r"\$message|\${message}", self._fmt)
            else:
                match = re.search(r"{message(?:![rsa])?(?:(?:.?[<>=^])?[+\- ]?z?#?0?\d*[_,]?(?:\.\d+)?[bcdeEfFgGnosxX%])?}", self._fmt)
            if match is None:
                indent = 0
            else:
                self._style._fmt = self._fmt[: match.start()]
                record.message = record.getMessage()
                if self.usesTime():
                    record.asctime = super().formatTime(record, self.datefmt)
                string = super().formatMessage(record)
                self._style._fmt = self._fmt
                indent = len(string)
        else:
            indent = 0
        # Format as normal
        string = super().format(record)
        string = string.replace("\n", "\n" + " " * indent)
        return string


class ColoringIndentingFormatter(IndentingFormatter):
    @override
    def format(self, record: logging.LogRecord) -> str:
        # Format with indenting
        string = super().format(record)
        # Colorize by level
        if record.levelno >= logging.CRITICAL:
            color_code = "\x1b[1;4;91m"
        elif record.levelno >= logging.ERROR:
            color_code = "\x1b[31m"
        elif record.levelno >= logging.WARNING:
            color_code = "\x1b[33m"
        elif record.levelno >= logging.INFO:
            color_code = "\x1b[0m"
        else:
            color_code = "\x1b[90m"
        string = color_code + string + "\x1b[0m"
        return string
