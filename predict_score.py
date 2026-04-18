import os
import csv
import calc_math
import random
import numpy as np
import math

import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import pickle

try:
    from scipy.stats import pearsonr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from utils import load_model

# ============================================================
# 校准/区间配置
# ============================================================
CALIBRATOR_TYPE = "isotonic"  # "isotonic" | "monotonic_mlp"
CONFORMAL_ALPHA = 0.10        # 0.10 -> 90% 区间；0.05 -> 95%
CONDITIONAL_BINS = 5          # 条件区间分箱数（按预测分位数切分）
GROUP_CALIBRATION = True      # True -> 按 raw 分位数(低/中/高)分组拟合校准器
SAVE_CALIBRATORS = True       # 保存校准器与区间参数，便于复现/部署
CALIBRATOR_OUT = os.path.join("model", "score_calibrators.pkl")



# ============================================================
# 期末题 -> 知识点 映射（9个知识点；第28题改成知识点9，其它不变）
# ============================================================
EXAM_QID_TO_PID = {
    **{i: 1 for i in range(1, 5)},
    **{i: 2 for i in range(5, 15)},
    15: 3,
    **{i: 4 for i in [16, 17, 19, 31]},
    18: 5,
    **{i: 6 for i in [20, 21, 22, 32, 33]},
    **{i: 7 for i in [23, 24, 34]},
    **{i: 8 for i in list(range(25, 31)) + [35]},
}
EXAM_QID_TO_PID[28] = 9  # ✅ 新增知识点9：第28题属于9


def exam_item_score(exam_qid: int) -> int:
    """前30题每题2分，后5题每题8分"""
    return 2 if 1 <= exam_qid <= 30 else 8


# ============================================================
# 读取数据：4行格式（id / qid / pid / ans）
# ============================================================
def read_student_sequences_4line(path: str, sep=","):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip().replace('"', "") for l in f.readlines() if l.strip() != ""]
    assert len(lines) % 4 == 0, f"{path} 行数必须为4的倍数"

    for i in range(0, len(lines), 4):
        sid = lines[i].strip()
        q_seq = [int(x) for x in lines[i + 1].split(sep) if x != ""]
        p_seq = [int(x) for x in lines[i + 2].split(sep) if x != ""]
        a_seq = [int(x) for x in lines[i + 3].split(sep) if x != ""]

        assert len(q_seq) == len(p_seq) == len(a_seq), f"{sid} 序列长度不一致"

        # 同一学生多段：直接拼接
        if sid not in data:
            data[sid] = (q_seq, p_seq, a_seq)
        else:
            q0, p0, a0 = data[sid]
            data[sid] = (q0 + q_seq, p0 + p_seq, a0 + a_seq)

    return data


def merge_three_sets(data_dir: str):
    train_path = os.path.join(data_dir, "my_data_train.csv")
    valid_path = os.path.join(data_dir, "my_data_valid.csv")
    test_path  = os.path.join(data_dir, "my_data_test.csv")

    d1 = read_student_sequences_4line(train_path)
    d2 = read_student_sequences_4line(valid_path)
    d3 = read_student_sequences_4line(test_path)

    merged = {}
    for d in [d1, d2, d3]:
        for sid, (q, p, a) in d.items():
            if sid not in merged:
                merged[sid] = (q, p, a)
            else:
                q0, p0, a0 = merged[sid]
                merged[sid] = (q0 + q, p0 + p, a0 + a)

    return merged


# ============================================================
# 读取期末真实成绩：exam.csv 两行一组（id / score）
# ============================================================
def read_exam_scores(path: str):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip().replace('"', "") for l in f.readlines() if l.strip() != ""]
    assert len(lines) % 2 == 0, "exam.csv 行数必须为2的倍数"

    for i in range(0, len(lines), 2):
        sid = lines[i].strip()
        sc = float(lines[i + 1].strip())
        scores[sid] = sc
    return scores


# ============================================================
# 从checkpoint推断模型结构，避免load_state_dict size mismatch
# ============================================================
def infer_model_hparams_from_state_dict(sd):
    # n_question/n_pid/d_model
    n_question = sd["q_embed.weight"].shape[0] - 1
    n_pid = sd["difficult_param.weight"].shape[0] - 1
    d_model = sd["q_embed.weight"].shape[1]

    # 推断 n_block：找 blocks.{i}.xxx 最大 i
    max_block = -1
    for k in sd.keys():
        if k.startswith("blocks."):
            # blocks.{i}.something
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                max_block = max(max_block, int(parts[1]))
    n_block = max_block + 1 if max_block >= 0 else 1

    return n_question, n_pid, d_model, n_block


# ============================================================
# 统一ID映射：复用训练阶段保存的 model/id_maps.json
# 避免“训练/推理 QID 编码不一致”导致的结果漂移。
# ============================================================
def load_id_maps(path: str = os.path.join("model", "id_maps.json")):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    qid_map_raw = obj.get("qid_map", {})
    pid_map_raw = obj.get("pid_map", {})

    def _to_int_dict(d):
        out = {}
        for k, v in d.items():
            try:
                out[int(k)] = int(v)
            except Exception:
                continue
        return out

    return {"qid_map": _to_int_dict(qid_map_raw), "pid_map": _to_int_dict(pid_map_raw)}


def make_qid_mapper_from_saved_maps(id_maps, unknown_policy: str = "zero"):
    """unknown_policy: 'zero' | 'hash'. 论文/部署建议用 'zero'。"""
    qid_map = (id_maps or {}).get("qid_map", {})

    def map_qid(q, n_question=None):
        q = int(q)
        if q <= 0:
            return 0
        if q in qid_map:
            return int(qid_map[q])
        if unknown_policy == "hash" and n_question is not None and n_question > 0:
            return (abs(hash(q)) % int(n_question)) + 1
        return 0

    return map_qid


# ============================================================
# QID 映射：把原始大QID映射到模型可用范围 1..n_question
# - 若唯一QID数量 <= n_question：按排序编号映射（信息保留最好）
# - 若唯一QID数量 > n_question：超出的用稳定hash映射到范围内（保证可跑）
# ============================================================
def build_qid_mapper(all_students, n_question):
    uniq = set()
    for sid, (q_seq, p_seq, a_seq) in all_students.items():
        for q in q_seq:
            if q > 0:
                uniq.add(int(q))

    uniq_sorted = sorted(list(uniq))
    mapping = {}

    if len(uniq_sorted) <= n_question:
        for idx, q in enumerate(uniq_sorted, start=1):
            mapping[q] = idx
        mode = "rank_map"
    else:
        # 先映射前 n_question 个最常见/最小的（这里用最小的，稳定）
        base = uniq_sorted[:n_question]
        for idx, q in enumerate(base, start=1):
            mapping[q] = idx
        mode = "hybrid_hash"

    def map_qid(q):
        q = int(q)
        if q <= 0:
            return 0
        if q in mapping:
            return mapping[q]
        # 稳定hash到 1..n_question
        return (abs(hash(q)) % n_question) + 1

    return map_qid, mode, len(uniq_sorted)


# ============================================================
# 每个PID选择锚题：用学生历史最后一次出现的(映射后qid)，否则用全局PID锚题
# ============================================================
def build_global_anchor_by_pid(all_students, map_qid):
    pid_to_qids = {}
    for sid, (q_seq, p_seq, a_seq) in all_students.items():
        for q, p in zip(q_seq, p_seq):
            q = map_qid(q)
            p = int(p)
            if q <= 0 or p <= 0:
                continue
            pid_to_qids.setdefault(p, set()).add(q)

    pid_anchor = {}
    for p, qs in pid_to_qids.items():
        if len(qs) > 0:
            pid_anchor[p] = min(qs)
    return pid_anchor


@torch.no_grad()
def predict_prob_for_pid(
    model,
    seqlen,
    n_question,
    n_pid,
    q_seq, p_seq, a_seq,
    target_pid,
    anchor_qid,
    map_qid,
    fallback_prob=0.5
):
    """
    用学生历史上下文（最近seqlen-1步），末尾追加一个(anchor_qid, target_pid, a=0)，
    取最后一个位置的pred作为该PID掌握概率。
    """
    target_pid = int(target_pid)
    if target_pid <= 0:
        return fallback_prob, "fallback_pid<=0"
    if target_pid > n_pid:
        # 超出模型支持范围就夹到最后
        target_pid = n_pid

    anchor_qid = map_qid(anchor_qid)
    if anchor_qid <= 0:
        anchor_qid = 1

    ctx_len = seqlen - 1
    q_ctx = q_seq[-ctx_len:] if len(q_seq) > ctx_len else q_seq[:]
    p_ctx = p_seq[-ctx_len:] if len(p_seq) > ctx_len else p_seq[:]
    a_ctx = a_seq[-ctx_len:] if len(a_seq) > ctx_len else a_seq[:]

    # 映射/裁剪
    q_ctx = [map_qid(x) for x in q_ctx]
    p_ctx = [min(max(int(x), 0), n_pid) for x in p_ctx]
    a_ctx = [1 if int(x) == 1 else 0 for x in a_ctx]

    q_arr = [0] * seqlen
    p_arr = [0] * seqlen
    qa_arr = [0] * seqlen

    L = len(q_ctx)
    for i in range(L):
        q_arr[i] = q_ctx[i]
        p_arr[i] = p_ctx[i]
        qa_arr[i] = q_ctx[i] + a_ctx[i] * n_question  # 与训练一致的编码

    last = L if L < seqlen else seqlen - 1
    q_arr[last] = anchor_qid
    p_arr[last] = target_pid
    qa_arr[last] = anchor_qid  # a=0 => qa=q

    q_t = torch.LongTensor([q_arr]).to(next(model.parameters()).device)
    qa_t = torch.LongTensor([qa_arr]).to(next(model.parameters()).device)
    pid_t = torch.LongTensor([p_arr]).to(next(model.parameters()).device)

    pred = model(q_t, qa_t, pid_t)  # [1, seqlen]
    prob = float(pred[0, last].item())
    return prob, "ok"


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))
    var_err = float(np.var(y_pred - y_true))

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else 0.0
    if _HAS_SCIPY and len(y_true) >= 2:
        try:
            pr, _ = pearsonr(y_true, y_pred)
            pr = float(pr)
        except Exception:
            pr = 0.0
    else:
        pr = 0.0

    return mae, rmse, bias, var_err, r2, pr


# ============================================================
# 校准器：isotonic / 单调 MLP（可微）
# ============================================================
class _MonotonicMLP(torch.nn.Module):
    """一个轻量的可微单调校准层。

    目标：学习一个单调函数 f(x)，把 raw score 映射到 calibrated score。
    实现：用若干个 ReLU “折线基函数”叠加，并通过 softplus 保证系数非负，从而保证单调。

    f(x) = b + a*x + sum_j softplus(w_j) * relu(x - t_j)
    其中 a>=0, w_j>=0。
    """

    def __init__(self, n_knots: int = 20):
        super().__init__()
        self.n_knots = int(n_knots)
        # knot 位置 t_j（可学习）；初始化为均匀分布
        self.t = torch.nn.Parameter(torch.linspace(0.0, 100.0, steps=self.n_knots))
        # 线性项 a 与偏置 b
        self.a_raw = torch.nn.Parameter(torch.tensor(0.1))
        self.b = torch.nn.Parameter(torch.tensor(0.0))
        # 每个 knot 的系数 w_j
        self.w_raw = torch.nn.Parameter(torch.zeros(self.n_knots))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,)
        x = x.view(-1)
        a = torch.nn.functional.softplus(self.a_raw)
        w = torch.nn.functional.softplus(self.w_raw)
        # relu(x - t_j)
        # (N, K) = (N,1) - (K,) broadcast
        relu_terms = torch.relu(x.unsqueeze(1) - self.t.unsqueeze(0))
        y = self.b + a * x + torch.sum(w.unsqueeze(0) * relu_terms, dim=1)
        return y


def fit_monotonic_mlp(x_cal: np.ndarray, y_cal: np.ndarray, n_knots: int = 20,
                      lr: float = 5e-2, steps: int = 2000, seed: int = 2024):
    """在校准集上训练可微单调校准层（分阶段训练即可，避免改动AKT训练流程）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _MonotonicMLP(n_knots=n_knots)
    model.train()

    x = torch.tensor(x_cal, dtype=torch.float32)
    y = torch.tensor(y_cal, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Huber 更稳健
    huber = torch.nn.SmoothL1Loss(beta=2.0)

    for _ in range(int(steps)):
        opt.zero_grad()
        pred = model(x)
        loss = huber(pred, y)
        loss.backward()
        opt.step()

    model.eval()
    return model


def predict_with_calibrator(calib, x: np.ndarray):
    """统一预测接口：支持 IsotonicRegression 或 _MonotonicMLP。"""
    if isinstance(calib, IsotonicRegression):
        return calib.predict(x)
    if isinstance(calib, _MonotonicMLP):
        with torch.no_grad():
            xx = torch.tensor(x, dtype=torch.float32)
            yy = calib(xx).cpu().numpy()
        return yy
    raise TypeError(f"Unsupported calibrator type: {type(calib)}")


def quantile_bins(values: np.ndarray, n_bins: int):
    """按分位数切分，返回 bin_edges（长度 n_bins+1）。"""
    values = np.asarray(values, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    # 处理重复边界
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges


def assign_bins(values: np.ndarray, bin_edges: np.ndarray):
    # 返回 0...(n_bins-1)
    return np.digitize(values, bin_edges[1:-1], right=False)


def conformal_q(residuals: np.ndarray, alpha: float):
    residuals = np.asarray(residuals, dtype=float)
    return float(np.quantile(residuals, 1.0 - float(alpha)))


def main():
    random.seed(2024)
    np.random.seed(2024)

    data_dir = "data/my_data"
    exam_path = os.path.join(data_dir, "exam.csv")
    model_path = os.path.join("model", "best_model.pt")

    # 1) 读三集合学生序列
    all_students = merge_three_sets(data_dir)

    # 2) 读真实分数
    exam_scores = read_exam_scores(exam_path)

    # 3) load checkpoint & 推断结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(model_path, map_location=device)

    n_question, n_pid, d_model, n_block = infer_model_hparams_from_state_dict(sd)

    # 4) 构造params并建模（保证与checkpoint匹配）
    class P:
        pass

    params = P()
    params.n_question = int(n_question)
    params.n_pid = int(n_pid)
    params.d_model = int(d_model)
    params.n_block = int(n_block)
    params.dropout = 0.1
    params.kq_same = 1
    params.l2 = 1e-5

    model = load_model(params).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    seqlen = 200

    print(f"[CKPT] n_question={n_question}, n_pid={n_pid}, d_model={d_model}, n_block={n_block}")
    print(f"[Predict] seqlen={seqlen}")

    # 5) 建 QID 映射器：优先复用训练阶段保存的 id_maps.json（强烈推荐）
    id_maps = load_id_maps(os.path.join("model", "id_maps.json"))
    if id_maps is not None and len(id_maps.get("qid_map", {})) > 0:
        map_qid = make_qid_mapper_from_saved_maps(id_maps, unknown_policy="zero")
        print(f"[QID MAP] mode=saved_id_maps, mapped_range=0..{n_question}")
    else:
        map_qid, map_mode, uniq_cnt = build_qid_mapper(all_students, n_question=n_question)
        print(f"[QID MAP] mode={map_mode}, unique_qid={uniq_cnt}, mapped_range=1..{n_question}")

    # 6) 全局每个PID锚题
    global_anchor = build_global_anchor_by_pid(all_students, map_qid=map_qid)

    # 7) 逐学生预测期末 35 题期望分
    out_rows = []
    matched = 0

    for sid, (q_seq, p_seq, a_seq) in all_students.items():
        # --- 🌟 修复点 1：不踢除没成绩的人，打上 NaN 标记 ---
        if sid in exam_scores:
            true_exam = float(exam_scores[sid])
            matched += 1
            if matched % 10 == 0:
                print(f"⏳ 正在疯狂计算中... 已完成 {matched} 名学生的期末 35 题预测", flush=True)
        else:
            true_exam = np.nan

            # 学生自己的 PID->anchor（取最后一次出现的映射后QID）
        student_anchor = {}
        for q, p in zip(q_seq, p_seq):
            q_m = map_qid(q)
            p = int(p)
            if q_m > 0 and p > 0:
                student_anchor[p] = q_m

        total_expected = 0.0
        row = {"student_id": sid, "true_exam": true_exam}

        for exam_qid in range(1, 36):
            pid = EXAM_QID_TO_PID[exam_qid]
            score = exam_item_score(exam_qid)

            anchor = student_anchor.get(pid, global_anchor.get(pid, 1))

            prob, _ = predict_prob_for_pid(
                model=model,
                seqlen=seqlen,
                n_question=n_question,
                n_pid=n_pid,
                q_seq=q_seq,
                p_seq=p_seq,
                a_seq=a_seq,
                target_pid=pid,
                anchor_qid=anchor,
                map_qid=map_qid,
                fallback_prob=0.5
            )

            exp_pts = prob * score
            total_expected += exp_pts

            row[f"q{exam_qid}_prob"] = prob
            row[f"q{exam_qid}_expected_pts"] = exp_pts

        row["raw_expected_total"] = total_expected
        out_rows.append(row)

    print(f"匹配总人数：{matched}")
    if matched == 0:
        print("没有任何学生能与 exam.csv 匹配，请检查 student_id 是否一致。")
        return

    # 8) 输出 raw expected
    out_csv = "exam_predictions_summary_expected_raw.csv"
    fieldnames = list(out_rows[0].keys())
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"已输出：{out_csv}")

    # --- 🌟 修复点 2：只挑选有真实成绩的人去训练模型 ---
    valid_out_rows = [r for r in out_rows if not np.isnan(r["true_exam"])]
    ids = [r["student_id"] for r in valid_out_rows]

    random.shuffle(ids)
    n = len(ids)
    n_cal = int(round(n * 0.8))
    cal_ids = set(ids[:n_cal])
    test_ids = set(ids[n_cal:])

    cal_rows = [r for r in valid_out_rows if r["student_id"] in cal_ids]
    test_rows = [r for r in valid_out_rows if r["student_id"] in test_ids]
    print(f"参与校准人数：{n} | 校准(80%)={len(cal_rows)} | 测试(20%)={len(test_rows)}")

    x_cal = np.array([r["raw_expected_total"] for r in cal_rows], dtype=float)
    y_cal = np.array([r["true_exam"] for r in cal_rows], dtype=float)

    x_test = np.array([r["raw_expected_total"] for r in test_rows], dtype=float)
    y_test = np.array([r["true_exam"] for r in test_rows], dtype=float)

    # 10) 校准器拟合：全局 +（可选）分组校准
    def _fit_one_calibrator(xc, yc):
        if CALIBRATOR_TYPE == "isotonic":
            c = IsotonicRegression(out_of_bounds="clip")
            c.fit(xc, yc)
            return c
        if CALIBRATOR_TYPE == "monotonic_mlp":
            return fit_monotonic_mlp(xc, yc, n_knots=20, lr=5e-2, steps=2000)
        raise ValueError(f"Unknown CALIBRATOR_TYPE={CALIBRATOR_TYPE}")

    # --------- 分组定义：按 raw 分位数（低/中/高） ---------
    group_edges = None

    def _assign_group(x):
        if group_edges is None:
            return 0
        return int(np.digitize([x], group_edges[1:-1])[0])

    calibrators = {}
    global_calib = _fit_one_calibrator(x_cal, y_cal)
    calibrators["global"] = global_calib

    if GROUP_CALIBRATION:
        group_edges = quantile_bins(x_cal, n_bins=3)
        for g in range(3):
            idx = [i for i, r in enumerate(cal_rows) if _assign_group(float(r["raw_expected_total"])) == g]
            if len(idx) < 20:
                calibrators[f"group_{g}"] = global_calib
                continue
            xc = x_cal[idx]
            yc = y_cal[idx]
            calibrators[f"group_{g}"] = _fit_one_calibrator(xc, yc)

    # --------- 测试集点预测 ---------
    y_pred_raw = x_test

    def _predict_point(rows, x_arr):
        preds = []
        groups = []
        for r, x in zip(rows, x_arr):
            g = _assign_group(float(r["raw_expected_total"]))
            groups.append(g)
            calib = calibrators.get(f"group_{g}", global_calib) if GROUP_CALIBRATION else global_calib
            preds.append(float(predict_with_calibrator(calib, np.array([x], dtype=float))[0]))
        return np.array(preds, dtype=float), np.array(groups, dtype=int)

    yhat_cal_point, g_cal = _predict_point(cal_rows, x_cal)
    yhat_test_point, g_test = _predict_point(test_rows, x_test)

    # --- 🌟 修复点 3：把找出的规律套用到全班所有人身上！ ---
    x_all = np.array([r["raw_expected_total"] for r in out_rows], dtype=float)
    yhat_all_point, g_all = _predict_point(out_rows, x_all)

    # clip 到 0~100
    yhat_cal_point = np.clip(yhat_cal_point, 0.0, 100.0)
    yhat_test_point = np.clip(yhat_test_point, 0.0, 100.0)
    yhat_all_point = np.clip(yhat_all_point, 0.0, 100.0)

    mae0, rmse0, bias0, var0, r20, pr0 = compute_metrics(y_test, y_pred_raw)
    mae1, rmse1, bias1, var1, r21, pr1 = compute_metrics(y_test, yhat_test_point)

    print("\n===== 测试集：未校准（raw expected_total）=====")
    print(f"mae: {mae0:.6f}")
    print(f"rmse: {rmse0:.6f}")
    print(f"bias: {bias0:.6f}")
    print(f"var_error: {var0:.6f}")
    print(f"r2: {r20:.6f}")
    print(f"pearson_r: {pr0:.6f}")

    print(f"\n===== 测试集：校准后（{CALIBRATOR_TYPE}，clip 0~100）=====")
    print(f"mae: {mae1:.6f}")
    print(f"rmse: {rmse1:.6f}")
    print(f"bias: {bias1:.6f}")
    print(f"var_error: {var1:.6f}")
    print(f"r2: {r21:.6f}")
    print(f"pearson_r: {pr1:.6f}")

    # 11) Conformal 预测区间（全局 + 条件分箱；可选分组）
    resid_cal = np.abs(y_cal - yhat_cal_point)
    q_global = float(np.quantile(resid_cal, 1.0 - CONFORMAL_ALPHA))

    bin_edges = quantile_bins(yhat_cal_point, n_bins=int(CONDITIONAL_BINS))
    b_cal = assign_bins(yhat_cal_point, bin_edges)
    q_by_bin = {}
    for b in range(int(CONDITIONAL_BINS)):
        rr = resid_cal[b_cal == b]
        if len(rr) < 10:
            q_by_bin[b] = q_global
        else:
            q_by_bin[b] = float(np.quantile(rr, 1.0 - CONFORMAL_ALPHA))

    b_test = assign_bins(yhat_test_point, bin_edges)
    q_test_bin = np.array([q_by_bin.get(int(b), q_global) for b in b_test], dtype=float)

    # 给所有人的预测结果加区间
    b_all = assign_bins(yhat_all_point, bin_edges)
    q_all_bin = np.array([q_by_bin.get(int(b), q_global) for b in b_all], dtype=float)

    low_global = np.clip(yhat_test_point - q_global, 0.0, 100.0)
    high_global = np.clip(yhat_test_point + q_global, 0.0, 100.0)
    low_bin = np.clip(yhat_test_point - q_test_bin, 0.0, 100.0)
    high_bin = np.clip(yhat_test_point + q_test_bin, 0.0, 100.0)

    low_global_all = np.clip(yhat_all_point - q_global, 0.0, 100.0)
    high_global_all = np.clip(yhat_all_point + q_global, 0.0, 100.0)
    low_bin_all = np.clip(yhat_all_point - q_all_bin, 0.0, 100.0)
    high_bin_all = np.clip(yhat_all_point + q_all_bin, 0.0, 100.0)

    # 覆盖率/区间宽度（测试集）
    def _picp(y, lo, hi):
        y = np.asarray(y, dtype=float)
        return float(np.mean((y >= lo) & (y <= hi)))

    def _mpiw(lo, hi):
        return float(np.mean(np.asarray(hi) - np.asarray(lo)))

    picp_g = _picp(y_test, low_global, high_global)
    mpiw_g = _mpiw(low_global, high_global)
    picp_b = _picp(y_test, low_bin, high_bin)
    mpiw_b = _mpiw(low_bin, high_bin)

    print(f"\n===== Conformal 区间（alpha={CONFORMAL_ALPHA:.2f}）=====")
    print(f"Global interval: PICP={picp_g:.4f} | MPIW={mpiw_g:.4f} | q={q_global:.4f}")
    print(f"Binned interval: PICP={picp_b:.4f} | MPIW={mpiw_b:.4f} | bins={CONDITIONAL_BINS}")

    # 12) 输出所有人的最终结果！
    out_test_csv = "test_predictions_calibrated_conformal.csv"
    with open(out_test_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "student_id", "true_exam", "raw_expected_total",
            "pred_calibrated", "group", "bin",
            "low_global", "high_global",
            f"low_bin_{int((1 - CONFORMAL_ALPHA) * 100)}", f"high_bin_{int((1 - CONFORMAL_ALPHA) * 100)}"
        ])
        for r, pred, g, b, lo_g, hi_g, lo_b, hi_b in zip(
                out_rows, yhat_all_point, g_all, b_all, low_global_all, high_global_all, low_bin_all, high_bin_all
        ):
            # 没成绩的人，在最终表格里写入空白
            true_score_out = r["true_exam"] if not np.isnan(r["true_exam"]) else ""

            w.writerow([
                r["student_id"], true_score_out, float(r["raw_expected_total"]),
                float(pred), int(g), int(b),
                float(lo_g), float(hi_g),
                float(lo_b), float(hi_b)
            ])
    print(f"\n✅ 已成功为全班所有人生成预测，输出至：{out_test_csv}")

    # 13) 保存校准器与区间参数（复现/部署）
    if SAVE_CALIBRATORS:
        os.makedirs("model", exist_ok=True)
        payload = {
            "calibrator_type": CALIBRATOR_TYPE,
            "conformal_alpha": CONFORMAL_ALPHA,
            "conditional_bins": int(CONDITIONAL_BINS),
            "group_calibration": bool(GROUP_CALIBRATION),
            "group_edges": None if group_edges is None else group_edges.tolist(),
            "bin_edges": bin_edges.tolist(),
            "q_global": float(q_global),
            "q_by_bin": {int(k): float(v) for k, v in q_by_bin.items()},
            "calibrators": calibrators,
        }
        with open(CALIBRATOR_OUT, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved calibrators to: {CALIBRATOR_OUT}")


if __name__ == "__main__":
    main()