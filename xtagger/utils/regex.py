class EnglishRegExTagger(object):
    def __init__(self, use_default = True, *args):
        self.use_default = use_default
        if args is not None:
            self.patterns = []
            for pattern in args:
                if type(pattern) != tuple:
                    raise TypeError("Matching must be tuple")
                self.patterns.append(pattern)
        if use_default:
            self.default = [
                (r'.*ing$', 'VERB'),
                (r'.*ed$', 'VERB'),
                (r'.*es$', 'VERB'),
                (r'.*\'s$', 'NOUN'),
                (r'.*s$', 'NOUN'),
                (r'\*T?\*?-[0-9]+$', 'X'),
                (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'),
                (r'.*', 'NOUN')
            ]

            if hasattr(self,"patterns") and self.patterns:
                for pattern in self.patterns:
                    self.default.append(pattern)

    def get_patterns(self):
        if self.use_default:
            return self.default
        else:
            return self.patterns

