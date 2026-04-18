import csv
import calc_math
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

try:
    from scipy.stats import pearsonr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


RAW_EXPECTED_CSV = "exam_predictions_summary_expected_raw.csv"
SEED = 2024
CLIP_MIN, CLIP_MAX = 0.0, 100.0

# 搜索粒度：越小越准但越慢（建议0.005或0.01）
STEP = 0.005


def exam_item_score(exam_qid: int) -> int:
    return 2 if 1 <= exam_qid <= 30 else 8


def metrics(y_true, y_pred):
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


def pred_score_by_threshold(row, t):
    total = 0.0
    for q in range(1, 36):
        p = float(row[f"q{q}_prob"])
        if p >= t:
            total += exam_item_score(q)
    return float(np.clip(total, CLIP_MIN, CLIP_MAX))


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    with open(RAW_EXPECTED_CSV, "r", encoding="utf-8-sig", newline="") as f:
        dr = csv.DictReader(f)
        rows = list(dr)

    # 80/20 split by student
    ids = [r["student_id"].strip() for r in rows]
    random.shuffle(ids)
    n = len(ids)
    n_cal = int(round(n * 0.8))
    cal_ids = set(ids[:n_cal])
    test_ids = set(ids[n_cal:])

    cal = [r for r in rows if r["student_id"].strip() in cal_ids]
    test = [r for r in rows if r["student_id"].strip() in test_ids]

    print(f"人数：{n} | 校准(80%)={len(cal)} | 测试(20%)={len(test)}")

    # grid search t
    best_t = None
    best_mae = 1e18

    ts = np.arange(0.0, 1.0 + 1e-9, STEP)
    y_cal = [float(r["true_exam"]) for r in cal]

    for t in ts:
        pred_cal = [pred_score_by_threshold(r, float(t)) for r in cal]
        mae = mean_absolute_error(y_cal, pred_cal)
        if mae < best_mae:
            best_mae = float(mae)
            best_t = float(t)

    # evaluate on test
    y_test = [float(r["true_exam"]) for r in test]
    pred_test = [pred_score_by_threshold(r, best_t) for r in test]
    m = metrics(y_test, pred_test)

    print("\n===== 最优阈值（在校准集上按 MAE 选择）=====")
    print(f"best_t = {best_t:.3f} | cal_mae = {best_mae:.6f}")

    print("\n===== 测试集指标（使用 best_t）=====")
    print(f"MAE={m[0]:.6f} | RMSE={m[1]:.6f} | Bias={m[2]:.6f} | VarErr={m[3]:.6f} | R2={m[4]:.6f} | PearsonR={m[5]:.6f}")

    # 输出测试集预测
    out_csv = "threshold_best_t_test_predictions.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "true_exam", "pred_threshold_score", "best_t"])
        for r, p in zip(test, pred_test):
            w.writerow([r["student_id"].strip(), float(r["true_exam"]), float(p), best_t])

    print(f"\n已输出：{out_csv}")


if __name__ == "__main__":
    main()