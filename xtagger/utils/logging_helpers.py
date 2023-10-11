import logging
import re
from typing import Callable, List

from tqdm.auto import tqdm


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.INFO) -> None:
        super().__init__(level)

    def emit(self, record: str) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def suppress_hf_logs() -> None:
    from transformers import logging as hflogging
    hflogging.set_verbosity_warning()



def set_global_logging_level(level: Callable = logging.ERROR, prefices: List[str] = [""]) -> None:
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)