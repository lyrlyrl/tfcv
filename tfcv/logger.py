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

class LogType:
    METRIC = 0
    PERF = 1
    FINALIZE = 2
    MESSAGE = 3

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
    def metric(self, data, step, verbosity=1):
        assert isinstance(data, Mapping)
        self.log(step, data, LogType.METRIC, verbosity)
    def perf(self, data, step, verbosity=1):
        assert isinstance(data, Mapping)
        self.log(step, data, LogType.PERF, verbosity)
    def finalize(self, data, step):
        self.log(step, data, LogType.FINALIZE, Verbosity.INFO)
    def message(self, data, step, verbosity=1):
        assert isinstance(data, str)
        self.log(step, data, LogType.MESSAGE, verbosity)
    def log(self, step, data, dtype, verbosity=1):
        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            if b.verbosity >= verbosity:
                b.log(timestamp, elapsedtime, step, dtype, data)
    

def default_step_format(step):
    return str(step)

def default_metric_format(metric,  value):
    format = "{}"
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

    def log(self, timestamp, elapsedtime, step, dtype, data):
        prefix = "{}{}".format(
            self.prefix_format(timestamp),
            self.step_format(step))
        if dtype == LogType.FINALIZE:
            if data == 'success':
                logs = data
            else:
                logs = 'failed: {}'.format(data)
        elif dtype == LogType.MESSAGE:
            logs = data
        else:
            assert isinstance(data, Mapping)
            if dtype == LogType.METRIC:
                logs = 'metrics: \n'
            elif dtype == LogType.PERF:
                logs = 'perf: \n'
            else:
                logs = '\n'
            for k, v in data.items():
                logs += '{}: {}\n'.format(k, v)
            
        print("{} {}".format(prefix, logs))

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
    def log(self, timestamp, elapsedtime, step, dtype, data):
        logs = dict(
                    timestamp=str(timestamp.timestamp()),
                    elapsedtime=str(elapsedtime),
                    datetime=str(timestamp),
                    type=dtype,
                    step=int(step),
                )
        if dtype == LogType.FINALIZE:
            if data == 'success':
                logs['success'] = True
            else:
                logs['success'] = False
                logs['additional_message'] = data
        elif dtype == LogType.MESSAGE:
            logs['data'] = data
        else:
            assert isinstance(data, Mapping)
            logs['data'] = data

        self.file.write('LOG {}\n'.format(
            json.dumps(logs)
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

def message(step, data, verbosity=Verbosity.INFO):
    GLOBAL_LOGGER.message(data, step, verbosity)

def finalize(step, data, verbosity=Verbosity.INFO):
    GLOBAL_LOGGER.finalize(data, step, verbosity)

def metric(step, data, verbosity=Verbosity.INFO):
    GLOBAL_LOGGER.metric(data, step, verbosity)

def perf(step, data, verbosity=Verbosity.INFO):
    GLOBAL_LOGGER.perf(data, step, verbosity)

def flush():
    GLOBAL_LOGGER.flush()

def init(backends):
    global GLOBAL_LOGGER
    try:
        if isinstance(GLOBAL_LOGGER, Logger):
            raise LoggerAlreadyInitialized()
    except LoggerNotInitialized:
        GLOBAL_LOGGER = Logger(backends)