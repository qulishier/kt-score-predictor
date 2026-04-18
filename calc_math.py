import os
import csv
import calc_math
import random
import numpy as np
import math

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import pearsonr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ========= 你需要改的路径（通常不用改） =========
RAW_EXPECTED_CSV = "exam_predictions_summary_expected_raw.csv"  # 模型二输出
MATH_CSV = os.path.join("data", "my_data", "math.csv")
OUT_CSV = "ability_score_predictions.csv"
SEED = 2024
CLIP_MIN, CLIP_MAX = 0.0, 100.0
CONFORMAL_ALPHA = 0.10
CONDITIONAL_BINS = 5
# ============================================


def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def read_math_scores(path: str):
    """
    支持两种格式：
      A) 带表头：student_id,math_score
      B) 两行一组：id / score
    返回 dict: sid -> float(math_score)
    """
    rows = _read_csv_rows(path)
    rows = [r for r in rows if len(r) > 0 and any(x.strip() for x in r)]

    # 可能是A格式
    if len(rows[0]) >= 2 and rows[0][0].strip().lower() in ("student_id", "id"):
        scores = {}
        for r in rows[1:]:
            if len(r) < 2:
                continue
            sid = r[0].strip()
            try:
                sc = float(r[1])
            except Exception:
                continue
            scores[sid] = sc
        return scores

    # 否则尝试B格式：两行一组（每行单列或多列拼起来）
    flat = []
    for r in rows:
        flat.append(",".join([x.strip() for x in r if x.strip() != ""]).strip())
    flat = [x for x in flat if x != ""]
    assert len(flat) % 2 == 0, "math.csv 不是表头格式时，必须两行一组"

    scores = {}
    for i in range(0, len(flat), 2):
        sid = flat[i].strip()
        sc = float(flat[i + 1].strip())
        scores[sid] = sc
    return scores


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


def quantile_bins(values: np.ndarray, n_bins: int):
    values = np.asarray(values, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges


def assign_bins(values: np.ndarray, bin_edges: np.ndarray):
    return np.digitize(values, bin_edges[1:-1], right=False)


def zscore(x, mu, sigma):
    return (x - mu) / (sigma + 1e-8)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # 读模型二输出：student_id,true_exam,raw_expected_total,...
    with open(RAW_EXPECTED_CSV, "r", encoding="utf-8-sig", newline="") as f:
        dr = csv.DictReader(f)
        rows = list(dr)

    math_scores = read_math_scores(MATH_CSV)

    # 组装数据
    data = []
    for r in rows:
        sid = r["student_id"].strip()
        y = float(r["true_exam"])
        x = float(r["raw_expected_total"])
        m = math_scores.get(sid, None)
        data.append((sid, x, y, m))

    # 80/20 split
    ids = [d[0] for d in data]
    random.shuffle(ids)
    n = len(ids)
    n_cal = int(round(n * 0.8))
    cal_ids = set(ids[:n_cal])
    test_ids = set(ids[n_cal:])

    cal = [d for d in data if d[0] in cal_ids]
    test = [d for d in data if d[0] in test_ids]

    print(f"人数：{n} | 训练(80%)={len(cal)} | 测试(20%)={len(test)}")
    cal_has_math = [d for d in cal if d[3] is not None]
    print(f"训练集中有高数的人数：{len(cal_has_math)}")

    # 标准化参数（用训练集）
    x_cal = np.array([d[1] for d in cal], dtype=float)
    mu_x, sd_x = float(np.mean(x_cal)), float(np.std(x_cal))

    # 高数只用有值的计算z
    m_cal_vals = np.array([d[3] for d in cal_has_math], dtype=float) if len(cal_has_math) > 0 else np.array([0.0])
    mu_m, sd_m = float(np.mean(m_cal_vals)), float(np.std(m_cal_vals)) if len(m_cal_vals) > 1 else 1.0

    # 训练线性融合权重（只在训练集中“有高数”的子集上拟合）
    # 目标：拟合 y （真实分） ~ w1*z_akt + w2*z_math + b
    if len(cal_has_math) >= 5:
        X = []
        Y = []
        for sid, x, y, m in cal_has_math:
            z_akt = zscore(x, mu_x, sd_x)
            z_math = zscore(float(m), mu_m, sd_m)
            X.append([z_akt, z_math])
            Y.append(y)
        lr = LinearRegression()
        lr.fit(np.array(X, dtype=float), np.array(Y, dtype=float))
        w1, w2 = float(lr.coef_[0]), float(lr.coef_[1])
        b = float(lr.intercept_)
    else:
        # 高数太少就退化：只用AKT
        w1, w2, b = 1.0, 0.0, 0.0

    print(f"ability权重：w1(z_akt)={w1:.6f}, w2(z_math)={w2:.6f}")

    # 构造 ability 分数（训练/测试都可以）
    def ability_score(x, m):
        z_akt = zscore(x, mu_x, sd_x)
        if m is None:
            z_math = 0.0
        else:
            z_math = zscore(float(m), mu_m, sd_m)
        return w1 * z_akt + w2 * z_math + b

    # 在训练集上 fit isotonic： ability -> y
    x_cal_ability = np.array([ability_score(d[1], d[3]) for d in cal], dtype=float)
    y_cal = np.array([d[2] for d in cal], dtype=float)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x_cal_ability, y_cal)

    # 测试集预测（点估计）
    test_y = np.array([d[2] for d in test], dtype=float)
    test_ability = np.array([ability_score(d[1], d[3]) for d in test], dtype=float)
    pred = np.clip(iso.predict(test_ability), CLIP_MIN, CLIP_MAX)

    # ===== Conformal 预测区间（分数域）=====
    yhat_cal = np.clip(iso.predict(x_cal_ability), CLIP_MIN, CLIP_MAX)
    resid_cal = np.abs(y_cal - yhat_cal)
    q_global = float(np.quantile(resid_cal, 1.0 - CONFORMAL_ALPHA))

    bin_edges = quantile_bins(yhat_cal, n_bins=int(CONDITIONAL_BINS))
    b_cal = assign_bins(yhat_cal, bin_edges)
    q_by_bin = {}
    for b in range(int(CONDITIONAL_BINS)):
        rr = resid_cal[b_cal == b]
        if len(rr) < 10:
            q_by_bin[b] = q_global
        else:
            q_by_bin[b] = float(np.quantile(rr, 1.0 - CONFORMAL_ALPHA))

    b_test = assign_bins(pred, bin_edges)
    q_test_bin = np.array([q_by_bin.get(int(b), q_global) for b in b_test], dtype=float)
    low_global = np.clip(pred - q_global, CLIP_MIN, CLIP_MAX)
    high_global = np.clip(pred + q_global, CLIP_MIN, CLIP_MAX)
    low_bin = np.clip(pred - q_test_bin, CLIP_MIN, CLIP_MAX)
    high_bin = np.clip(pred + q_test_bin, CLIP_MIN, CLIP_MAX)

    # 总体指标
    mae, rmse, bias, var_err, r2, pr = compute_metrics(test_y, pred)
    print("\n===== 测试集总体（ability -> isotonic）=====")
    print(f"MAE={mae:.6f} | RMSE={rmse:.6f} | Bias={bias:.6f} | VarErr={var_err:.6f} | R2={r2:.6f} | PearsonR={pr:.6f}")

    # 区间覆盖率/宽度
    picp_g = float(np.mean((test_y >= low_global) & (test_y <= high_global)))
    mpiw_g = float(np.mean(high_global - low_global))
    picp_b = float(np.mean((test_y >= low_bin) & (test_y <= high_bin)))
    mpiw_b = float(np.mean(high_bin - low_bin))
    print(f"Conformal(α={CONFORMAL_ALPHA:.2f}) Global: PICP={picp_g:.4f}, MPIW={mpiw_g:.4f}, q={q_global:.4f}")
    print(f"Conformal Binned: PICP={picp_b:.4f}, MPIW={mpiw_b:.4f}, bins={CONDITIONAL_BINS}")

    # 分有高数 / 无高数
    test_with = [(d, p) for d, p in zip(test, pred) if d[3] is not None]
    test_without = [(d, p) for d, p in zip(test, pred) if d[3] is None]

    if len(test_with) > 0:
        y = [dp[0][2] for dp in test_with]
        p = [dp[1] for dp in test_with]
        mae, rmse, bias, var_err, r2, pr = compute_metrics(y, p)
        print("\n----- 测试集：有高数 -----")
        print(f"MAE={mae:.6f} | RMSE={rmse:.6f} | Bias={bias:.6f} | VarErr={var_err:.6f} | R2={r2:.6f} | PearsonR={pr:.6f}")

    if len(test_without) > 0:
        y = [dp[0][2] for dp in test_without]
        p = [dp[1] for dp in test_without]
        mae, rmse, bias, var_err, r2, pr = compute_metrics(y, p)
        print("\n----- 测试集：无高数（只靠AKT贡献）-----")
        print(f"MAE={mae:.6f} | RMSE={rmse:.6f} | Bias={bias:.6f} | VarErr={var_err:.6f} | R2={r2:.6f} | PearsonR={pr:.6f}")

    # 输出预测文件（全体学生）：点估计 + 区间
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "student_id", "true_exam", "raw_expected_total", "math_score", "ability_raw",
            "pred_isotonic",
            "low_global", "high_global",
            f"low_bin_{int((1-CONFORMAL_ALPHA)*100)}", f"high_bin_{int((1-CONFORMAL_ALPHA)*100)}"
        ])
        for sid, x, y, m in data:
            a = ability_score(x, m)
            prd = float(np.clip(iso.predict([a])[0], CLIP_MIN, CLIP_MAX))
            # 区间：全局q + 条件bin
            b = int(assign_bins(np.array([prd]), bin_edges)[0])
            qb = float(q_by_bin.get(b, q_global))
            lo_g = float(np.clip(prd - q_global, CLIP_MIN, CLIP_MAX))
            hi_g = float(np.clip(prd + q_global, CLIP_MIN, CLIP_MAX))
            lo_b = float(np.clip(prd - qb, CLIP_MIN, CLIP_MAX))
            hi_b = float(np.clip(prd + qb, CLIP_MIN, CLIP_MAX))
            w.writerow([sid, y, x, "" if m is None else float(m), float(a), prd, lo_g, hi_g, lo_b, hi_b])

    print(f"\n已输出：{OUT_CSV}")


if __name__ == "__main__":
    main()