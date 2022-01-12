import atexit
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Mapping
import json

class Verbosity:
    OFF = -1
    DEBUG = 0
    AUTO = 1
    INFO = 1
    WARN = 2
    CRITICAL = 3

class Backend(ABC):
    def __init__(self, verbosity):
        self._verbosity = verbosity

    @property
    def verbosity(self):
        return self._verbosity

    @abstractmethod
    def log(self, timestamp, elapsedtime, step, data):
        pass
    

class Logger:
    def __init__(self, backends):
        self.backends = backends
        atexit.register(self.flush)
        self.starttime = datetime.now()
    def flush(self):
        for b in self.backends:
            b.flush()
    def metric(self, key, data, step=tuple()):
        self.log(step, f'{key}: {str(data)}')
    def metrics(self, data, step=tuple()):
        assert isinstance(data, Mapping)
        for k, v in data.items():
            self.metric(k, v, step)
    def finalize(self, success, step=tuple()):
        if success == True:
            self.log(step, {'status': 'success'})
        else:
            self.log(step, {'status': 'failed', 'callback': success})
    def log(self, step, data, verbosity=1):
        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            if b.verbosity >= verbosity:
                b.log(timestamp, elapsedtime, step, data)
    

def default_step_format(step):
    return str(step)

def default_metric_format(metric, metadata, value):
    format = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    return "{} : {} {}".format(
        metric, format.format(value)
    )

def default_prefix_format(timestamp):
    return "TFCV {} - ".format(timestamp)

class StdOutBackend(Backend):
    def __init__(
        self,
        verbosity,
        step_format=default_step_format,
        metric_format=default_metric_format,
        prefix_format=default_prefix_format,
    ):
        super().__init__(verbosity=verbosity)

        self.step_format = step_format
        self.metric_format = metric_format
        self.prefix_format = prefix_format

    def log(self, timestamp, elapsedtime, step, data):
        print(
            "{}{} {}".format(
                self.prefix_format(timestamp),
                self.step_format(step),
                " ".join(
                    [
                        self.metric_format(m, v)
                        for m, v in data.items()
                    ]
                ),
            )
        )

    def flush(self):
        pass

class FileBackend(Backend):
    def __init__(
        self, 
        verbosity,
        file_path,
        proceed=True):
        super().__init__(verbosity)
        self._file_path = file_path
        self.file = open(self._file_path, 'a' if proceed else 'w')
        atexit.register(self.file.close)
    def log(self, timestamp, elapsedtime, step, data):
        self.file.write('LOG {}\n'.format(
            json.dumps(
                dict(
                    timestamp=str(timestamp.timestamp()),
                    elapsedtime=str(elapsedtime),
                    datetime=str(timestamp),
                    type="LOG",
                    step=step,
                    data=data,
                )
            )
        ))
    def flush(self):
        self.file.flush()

class LoggerNotInitialized(Exception):
    pass

class LoggerAlreadyInitialized(Exception):
    pass

class NotInitializedObject(object):
    def __getattribute__(self, name):
        raise LoggerNotInitialized(
            "Logger not initialized. Initialize Logger with init(backends) function"
        )

GLOBAL_LOGGER = NotInitializedObject()

def log(step, data, verbosity=Verbosity.AUTO):
    GLOBAL_LOGGER.log(step, data, verbosity=verbosity)

def flush():
    GLOBAL_LOGGER.flush()

def init(backends):
    global GLOBAL_LOGGER
    try:
        if isinstance(GLOBAL_LOGGER, Logger):
            raise LoggerAlreadyInitialized()
    except LoggerNotInitialized:
        GLOBAL_LOGGER = Logger(backends)