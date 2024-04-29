import sys

class Vprint():
    verbose = False

    @classmethod
    def vprint(cls, *args:str, sep=' ', end='\n', file=sys.stdout, flush=False):
        if cls.verbose:
            print(*args, sep=sep, end=end, file=file, flush=flush)


vprint = Vprint.vprint
