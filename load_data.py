import numpy as np
import math
import calc_math


class PID_DATA(object):
    """
    每 4 行为一组：
    line 0: student_id（可忽略）
    line 1: QID 序列（题目）
    line 2: PID 序列（知识点）
    line 3: Answer 序列 (0/1)
    """

    def __init__(self, n_question, seqlen, separate_char=',', remap_ids=True):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question
        self.remap_ids = remap_ids

        # 原始ID -> 连续ID(从1开始)，0保留给padding
        self.qid_map = {}
        self.pid_map = {}

    def _map_id(self, x, mp):
        if x == 0:
            return 0
        if x not in mp:
            mp[x] = len(mp) + 1
        return mp[x]

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip().replace('"', '') for l in f.readlines()]

        assert len(lines) % 4 == 0, "数据行数必须是 4 的倍数"

        q_data, qa_data, p_data = [], [], []

        for i in range(0, len(lines), 4):
            # 关键：第二行是题目，第三行是知识点
            qid_seq = list(map(int, lines[i + 1].split(self.separate_char)))
            pid_seq = list(map(int, lines[i + 2].split(self.separate_char)))
            ans_seq = list(map(int, lines[i + 3].split(self.separate_char)))

            assert len(pid_seq) == len(qid_seq) == len(ans_seq)

            if self.remap_ids:
                qid_seq = [self._map_id(x, self.qid_map) for x in qid_seq]
                pid_seq = [self._map_id(x, self.pid_map) for x in pid_seq]

            n_split = math.ceil(len(qid_seq) / self.seqlen)

            for k in range(n_split):
                start = k * self.seqlen
                end = min((k + 1) * self.seqlen, len(qid_seq))

                q_tmp, qa_tmp, p_tmp = [], [], []

                for j in range(start, end):
                    q = qid_seq[j]
                    a = ans_seq[j]
                    p = pid_seq[j]

                    qa = q + a * self.n_question

                    q_tmp.append(q)
                    qa_tmp.append(qa)
                    p_tmp.append(p)

                q_data.append(q_tmp)
                qa_data.append(qa_tmp)
                p_data.append(p_tmp)

        q_arr = np.zeros((len(q_data), self.seqlen), dtype=np.int64)
        qa_arr = np.zeros((len(qa_data), self.seqlen), dtype=np.int64)
        p_arr = np.zeros((len(p_data), self.seqlen), dtype=np.int64)

        for i in range(len(q_data)):
            q_arr[i, :len(q_data[i])] = q_data[i]
            qa_arr[i, :len(qa_data[i])] = qa_data[i]
            p_arr[i, :len(p_data[i])] = p_data[i]

        return q_arr, qa_arr, p_arr
