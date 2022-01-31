class TermPair():
    
    def __init__(self, termA: str, termB: str):
        thisPair = [termA, termB]
        thisPair.sort()
        self._pair = thisPair