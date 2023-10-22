import io

class SequencedLog:
    _log_file: str
    _seq: int
    
    def __init__(self, log_file: str) -> None:
        self._log_file = log_file
        self._seq = 0

    def write_seq(self, *args):
        self.write(self._seq, *args)
        self._seq += 1

    def write(self, id: int, *args):
        file = open(self._log_file, "+a")
        strout = io.StringIO()
        strout.write(f"{id}")
        for p in args:
            strout.write(f"|{p}")
        file.write(f"{strout.getvalue()}\n")
        file.close()