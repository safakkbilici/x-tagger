import tqdm

class tqdmConfig(object):
    def __init__(self, notebook=True, disable = False):
        self.notebook = notebook
        self.disable = disable
