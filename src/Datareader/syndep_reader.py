__author__ = 'max'

import Zparser
tokens = Zparser


class DependencyInstance(object):
    def __init__(self, words, postags, heads, types):
        self.words = words
        self.postags = postags
        self.heads = heads
        self.types = types

    def length(self):
        return len(self.words)


class CoNLLXReader(object):
    def __init__(self, file_path, type_vocab = None):
        self.__source_file = open(file_path, 'r')
        self.type_vocab = type_vocab

    def close(self):
        self.__source_file.close()

    def getNext(self):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            #line = line.decode('utf-8')
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        postags = []
        types = []
        heads = []

        # words.append(Zparser.ROOT)
        # postags.append(Zparser.ROOT)
        # types.append(Zparser.ROOT)
        # heads.append(0)

        for tokens in lines:

            word = tokens[1]
            pos = tokens[4]
            head = int(tokens[6])
            type = tokens[7]

            words.append(word)

            postags.append(pos)

            types.append(type)

            heads.append(head)

        # words.append(parse_nk.STOP)
        # postags.append(parse_nk.STOP)
        # types.append(parse_nk.STOP)
        # heads.append(0)

        return DependencyInstance(words, postags, heads, types)


def read_syndep(syndep_path, max_len=0):

    dep_reader = CoNLLXReader(syndep_path)
    print('Reading dependency parsing data from %s' % syndep_path)

    counter = 0
    dep_sentences = []
    dep_pos = []
    dep_heads = []
    dep_types = []
    inst = dep_reader.getNext()
    while inst is not None:

        inst_size = inst.length()
        if max_len > 0 and inst_size - 1 > max_len:
            inst = dep_reader.getNext()
            continue

        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)
        dep_pos.append(inst.postags)
        # dep_sentences.append([(tag, word) for i, (word, tag) in enumerate(zip(sent.words, sent.postags))])
        dep_sentences.append(inst.words)
        dep_heads.append(inst.heads)
        dep_types.append(inst.types)
        inst = dep_reader.getNext()

    dep_reader.close()
    print("Total number of data: %d" % counter)

    return dep_sentences, dep_pos, dep_heads, dep_types