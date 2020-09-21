import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython

ctypedef np.float32_t DTYPE_t

_CORE_ARGS = {"A0": 1, "A1": 2, "A2": 4, "A3": 8, "A4": 16, "A5": 32, "AA": 64}
_CORE_ARGS_INDEX = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "AA": 6}
_CORE_ARGS_LIST = ["A0", "A1", "A2", "A3", "A4", "A5", "AA"]

@cython.boundscheck(False)
def decode(int start, int end, int leng, int verb_idx, np.ndarray[int, ndim=2] span_mat, np.ndarray[DTYPE_t, ndim=2] srl_score_np, srl_dpdict, srlspan_vocab):

    cdef DTYPE_t NEG_INF = -np.inf, span_score, max_score
    cdef int cun, left, right, state, span_idx, cur, cur_state, max_label_idx, label_idx
    cdef np.ndarray[DTYPE_t, ndim=1] max_srlscore = np.zeros(leng, dtype= np.float32)
    cdef np.ndarray[int, ndim=1] srl_path = np.zeros(leng, dtype= np.int32)

    for right in range(start + 1, end):
        max_score = -np.inf
        for left in range(start, right):
            span_idx = span_mat[left][right]
            if span_idx == -1:
                span_score = 0
            else:
                span_score = np.max(srl_score_np[span_idx])
            if span_score + max_srlscore[left] > max_score:
                max_score = span_score + max_srlscore[left]
                srl_path[right] = left

        max_srlscore[right] = max_score

    cur = end - 1
    while cur > start:
        left = srl_path[cur]
        right = cur
        span_idx = span_mat[left][right]
        if span_idx != -1:
            max_label_idx = np.argmax(srl_score_np[span_idx])
            if max_label_idx > 0 and srlspan_vocab.value(max_label_idx)[-1] != "*":
                if verb_idx not in srl_dpdict:
                    srl_dpdict[verb_idx] = []
                srl_dpdict[verb_idx].append((left, right - 1, srlspan_vocab.value(max_label_idx)))

        cur = left
