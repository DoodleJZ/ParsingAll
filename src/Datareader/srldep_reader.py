__author__ = 'max'

import Zparser
tokens = Zparser

class SrlDep(object):
    def __init__(self, words, pred_pos, gold_verb, gold_srl, syndep_heads):
        self.words = words
        self.pred_pos = pred_pos
        self.gold_verb = gold_verb
        self.gold_srl = gold_srl
        self.syndep_heads = syndep_heads

    def length(self):
        return len(self.words)


class CoNLLXReader(object):
    def __init__(self, file_path):
        self.__source_file = open(file_path, 'r')

    def close(self):
        self.__source_file.close()

    def getNext(self):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None, 0

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            #line = line.decode('utf-8')
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None, 0

        # verb id from 1

        words = []
        pred_pos = []
        syndep_heads = []
        gold_srl = {}
        gold_verb = []
        miss_verb = 0
        # words.append(Zparser.ROOT)
        # postags.append(Zparser.ROOT)
        # types.append(Zparser.ROOT)
        # heads.append(0)

        # srl begin in 10, verb is 8
        #if no verb, list is empty
        num_verb = len(lines[0]) - 10

        #create verb idx
        verb_id = []
        #vr is verb
        for i, tokens in enumerate(lines):

            label = tokens[8]
            if label == "Y":
                verb_id.append(i)
                gold_srl[i] = []
                gold_verb.append((i,i))

        if num_verb > len(verb_id) and len(verb_id) > 0:
            # print(num_verb - len(verb_id))
            miss_verb += 1
            num_verb = len(verb_id)

        for i, tokens in enumerate(lines):
            for v_id in range(num_verb):

                label = tokens[v_id + 10]
                if label != "_":
                    gold_srl[verb_id[v_id]].append((i, label))

            word = tokens[1]
            words.append(word)
            pred_pos.append(tokens[4])
            syndep_heads.append(int(tokens[6]))

        # words.append(parse_nk.STOP)
        # postags.append(parse_nk.STOP)
        # types.append(parse_nk.STOP)
        # heads.append(0)

        return SrlDep(words, pred_pos, gold_verb, gold_srl, syndep_heads), miss_verb


def read_srldep(srldep_path, max_len = 0):

    srlspan_reader = CoNLLXReader(srldep_path)
    print('Reading span srl data from %s' % srldep_path)
    counter = 0
    miss_verbs = 0
    words = []
    pred_pos = []
    gold_verb = []
    gold_srl = []
    syndep_heads = []
    srl_size = 0
    srl_inst, miss_verb = srlspan_reader.getNext()
    while srl_inst is not None:

        inst_size = srl_inst.length()
        miss_verbs += miss_verb
        if max_len > 0 and inst_size - 1 > max_len:
            srl_inst = srlspan_reader.getNext()
            continue

        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        words.append(srl_inst.words)
        pred_pos.append(srl_inst.pred_pos)
        gold_verb.append(srl_inst.gold_verb)
        gold_srl.append(srl_inst.gold_srl)
        syndep_heads.append(srl_inst.syndep_heads)
        for id, a in srl_inst.gold_srl.items():
            srl_size += len(a)
        if counter<0:
            print("sent id", counter)
            print(srl_inst.gold_srl)

        srl_inst, miss_verb = srlspan_reader.getNext()

    srlspan_reader.close()

    print("Total number of data: %d" % counter)
    print("miss verb", miss_verbs)
    print("srl size:", srl_size)
    print("=========================================")


    return words, pred_pos, gold_verb, gold_srl, syndep_heads