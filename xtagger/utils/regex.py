import re
class EnglishRegExTagger(object):
    def __init__(self, rules = None, use_default = True):
        self.use_default = use_default
        if rules is not None:
            self.patterns = []
            for pattern in rules:
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
                    if pattern not in self.default:
                        self.default.append(pattern)

    def get_patterns(self):
        if self.use_default:
            return self.default
        else:
            return self.patterns
        
    def tag(self, word):
        patterns = self.get_patterns()
        found = -1
        for rule in patterns:
            f = re.match(rule[0], word)
            if f != None:
                return rule[1]
        return found
