import time
import threading
import numpy as np

class DiscreteComponent:
    thr: threading.Thread
    _period_ms: int
    _run: bool
    _last_timestamp: float


    def __init__(self, period_ms: int) -> None:
        self._period_ms = period_ms / 1000
        self.thr = None
        self._run = False
        
    def start (self) -> None:
        if self._run:
            self.destroy()

        self._run = True
        self.thr = threading.Thread(target=self.__discrete_loop)
        self.thr.start()

    def destroy(self) -> None:
        self._run = False

        if self.thr is None:
            return
        
        self.thr = None

    def __discrete_loop(self):
        self._last_timestamp = time.time()
        
        while (self._run):
            t = time.time()
            dt = t - self._last_timestamp
            self._loop(dt)
            self._last_timestamp = t
            time.sleep(self._period_ms)


    def _loop(self, dt: float) -> None:
        pass
