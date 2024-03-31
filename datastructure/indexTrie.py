class indexTrie:
    def __init__(self):
        self.offset = -1
        self.children = None
    
    def set(self, val):
        self.offset = val
    
    def get(self):
        return self.offset