__author__ = 'max'

import Zparser
tokens = Zparser

class SrlInstance(object):
    def __init__(self, words, gold_verb, gold_srl, pred_pos, gold_pos, srlabel, srl_label_start, multi_verb, syndep_heads):
        self.words = words
        self.gold_verb = gold_verb
        self.gold_srl = gold_srl
        self.pred_pos = pred_pos
        self.gold_pos = gold_pos
        self.srlabel = srlabel
        self.srl_label_start = srl_label_start
        self.multi_verb = multi_verb
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

        # verb id from 1

        words = []
        pred_pos = []
        gold_srl = {}
        gold_pos = []
        gold_verb = []

        # words.append(Zparser.ROOT)
        # postags.append(Zparser.ROOT)
        # types.append(Zparser.ROOT)
        # heads.append(0)

        # srl begin in 14
        #if no verb, list is empty
        num_verb = len(lines[0]) - 14
        srl_type = []
        srl_label = []
        srl_label_start = []
        syndep_heads = []
        srl_start = []
        srl_end = []

        for i in range(num_verb):
            srl_type.append("*")
            srl_start.append(-1)

        #create verb idx
        verb_id = []
        multi_verb = 0

        for i, tokens in enumerate(lines):
            for v_id in range(num_verb):
                label = tokens[v_id + 14]

                label = label.replace("*", "")
                label = label.replace("(", "")
                label = label.replace(")","")
                label = label.strip()
                # if len(label)>0:
                #     print(label)

                if label == "V":
                    verb_id.append(i)
                    gold_srl[i] = []

        for i, tokens in enumerate(lines):
            srl_word = []
            srl_word_start = []
            for v_id in range(num_verb):

                label = tokens[v_id + 14]

                if len(label) >1:
                    label = label.replace("*", "")

                label = label.replace("(", " ( ").replace(")", " ) ").split()

                assert len(label) < 4

                if len(label) == 3:
                    assert label[0] == "(" and label[-1] == ")"

                if label[0] == "(" :
                    srl_start[v_id] = i
                    srl_type[v_id] = label[1]

                if len(label) < 2 and srl_type[v_id] == "V": #more than one verb, just regard as one verb
                    # gold_srl[verb_id[v_id]].append((srl_start[v_id], i + 1, srl_type[v_id]))#i+1 span[i,j] = w[j+1] - w[i]
                    # srl_type[v_id] = "*"
                    # srl_start[v_id] = -1
                    multi_verb += 1

                srl_word.append(srl_type[v_id])
                srl_word_start.append(srl_start[v_id])

                if label[-1] == ")":
                    if srl_type[v_id] != "*":
                        gold_srl[verb_id[v_id]].append((srl_start[v_id], i, srl_type[v_id])) #idx from 0
                        if srl_type[v_id] == "V":
                            gold_verb.append((srl_start[v_id], i))
                        srl_type[v_id] = "*"
                        srl_start[v_id] = -1

            word = tokens[3]
            pred_pos.append(tokens[5])
            gold_pos.append(tokens[4])

            words.append(word)
            syndep_heads.append(int(tokens[6]))

            # print(word, srl_word)

            srl_label.append(srl_word)
            srl_label_start.append(srl_word_start)

        # words.append(parse_nk.STOP)
        # postags.append(parse_nk.STOP)
        # types.append(parse_nk.STOP)
        # heads.append(0)

        for i in range(len(gold_verb)):
            if i > 0:
                assert gold_verb[i][0] > gold_verb[i - 1][0]

        return SrlInstance(words, gold_verb, gold_srl, pred_pos, gold_pos, srl_label, srl_label_start, multi_verb, syndep_heads)


def read_srlspan(srlspan_path, max_len = 0):

    srlspan_reader = CoNLLXReader(srlspan_path)
    print('Reading span srl data from %s' % srlspan_path)
    counter = 0
    words = []
    gold_verb = []
    gold_srl = []
    pred_pos = []
    gold_pos = []
    syndep_heads = []
    srlspan_label = []
    srlspan_label_start = []

    adj_srl = 0
    srl_size = 0
    max_span = 0
    max_verb = 0
    multi_verb  = 0
    srl_inst = srlspan_reader.getNext()
    while srl_inst is not None:

        inst_size = srl_inst.length()
        if max_len > 0 and inst_size - 1 > max_len:
            srl_inst = srlspan_reader.getNext()
            continue

        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        words.append(srl_inst.words)
        gold_verb.append(srl_inst.gold_verb)
        gold_srl.append(srl_inst.gold_srl)
        pred_pos.append(srl_inst.pred_pos)
        gold_pos.append(srl_inst.gold_pos)
        syndep_heads.append(srl_inst.syndep_heads)
        srlspan_label.append(srl_inst.srlabel)
        srlspan_label_start.append(srl_inst.srl_label_start)
        multi_verb += srl_inst.multi_verb
        if counter<0:
            print("sent id", counter)
            print(srl_inst.gold_srl)
            print(srl_inst.srlabel)
            print(srl_inst.srl_label_start)

        max_verb = max(max_verb, len(srl_inst.gold_srl))
        inst_num_span = 0
        for id, a in srl_inst.gold_srl.items():
            srl_size += len(a)
            inst_num_span += len(a)

            b_f = -2
            b_l = ""
            for (left,right,srlabel) in a:
                if left == b_f + 1 and srlabel == b_l:
                    adj_srl += 1
                    #print(counter, srl_inst.gold_srl[id])
                b_f = right
                b_l = srlabel
        max_span = max(max_span, inst_num_span)
        srl_inst = srlspan_reader.getNext()

    srlspan_reader.close()

    print("Total number of data: %d" % counter)

    print("srl size:", srl_size)
    print("srl adj num:", adj_srl)
    print("max verb span num", max_verb, max_span)
    print("multi_verb:", multi_verb)
    print("=========================================")


    return words, gold_verb, gold_srl, pred_pos, gold_pos, srlspan_label, srlspan_label_start, syndep_heads