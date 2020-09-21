__author__ = 'max'

import re
import numpy as np


def eval(gold_pos, pred_pos):

    corr = 0.
    total = 0.
    assert len(gold_pos) == len(pred_pos)
    for golds, predictions in zip(gold_pos, pred_pos):
        assert len(golds) == len(predictions)
        for gold, prediction in zip(golds, predictions):
            if gold == prediction:
                corr += 1
            total += 1

    print('POS: corr: %d, total: %d, acc: %.4f%%' % (
        corr, total, corr * 100 / total))

    return corr * 100 / total

