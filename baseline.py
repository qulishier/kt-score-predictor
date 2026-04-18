import os
import numpy as np
import pandas as pd

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    bias = np.mean(err)
    varerr = np.var(err, ddof=0)
    denom = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - (np.sum((y_true - y_pred)**2) / denom) if denom > 1e-12 else np.nan
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pr = np.nan
    else:
        pr = np.corrcoef(y_true, y_pred)[0, 1]
    return dict(mae=mae, rmse=rmse, bias=bias, var_error=varerr, r2=r2, pearson_r=pr)

def main():
    RAW_SUMMARY = "exam_predictions_summary_expected_raw.csv"
    CAL_RATIO = 0.8

    # 阈值搜索范围（你之前 best_t 很高，所以从 0.5~0.99 足够）
    T_GRID = np.linspace(0.5, 0.99, 100)

    if not os.path.exists(RAW_SUMMARY):
        raise FileNotFoundError(f"找不到文件：{RAW_SUMMARY}（请把脚本放在生成该CSV的同目录）")

    df = pd.read_csv(RAW_SUMMARY)

    # === 真实分列：你的CSV里是 true_exam ===
    if "true_exam" not in df.columns:
        raise KeyError(f"找不到真实分列 true_exam。实际列前30个：{list(df.columns)[:30]}")
    y_true = df["true_exam"].to_numpy(dtype=float)

    # === 35题概率列：q1_prob ... q35_prob ===
    prob_cols = [f"q{i}_prob" for i in range(1, 36)]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少概率列：{missing[:10]}...（共{len(missing)}个）。"
                       f"请检查是否为 q{i}_prob 命名。")
    probs = df[prob_cols].to_numpy(dtype=float)

    # === 分值：默认前30题2分，后5题8分 ===
    score_vec = np.array([2]*30 + [8]*5, dtype=float)
    scores = np.tile(score_vec[None, :], (len(df), 1))

    # === 切分：前80%校准，后20%测试（与你现有流程一致）===
    n = len(df)
    n_cal = int(round(n * CAL_RATIO))
    cal_idx = np.arange(0, n_cal)
    test_idx = np.arange(n_cal, n)

    probs_cal, y_cal, scores_cal = probs[cal_idx], y_true[cal_idx], scores[cal_idx]
    probs_test, y_test, scores_test = probs[test_idx], y_true[test_idx], scores[test_idx]

    def threshold_score(pmat, smat, t):
        correct = (pmat > t).astype(float)
        return np.sum(correct * smat, axis=1)

    # === 搜索 best_t（校准集 MAE 最小）===
    best_t, best_mae = None, 1e18
    for t in T_GRID:
        pred_cal = threshold_score(probs_cal, scores_cal, t)
        m = metrics(y_cal, pred_cal)
        if m["mae"] < best_mae:
            best_mae = m["mae"]
            best_t = float(t)

    # === 测试集评估 ===
    pred_test = threshold_score(probs_test, scores_test, best_t)
    m_test = metrics(y_test, pred_test)

    print("===== 阈值法 baseline =====")
    print(f"校准集(80%) 搜索 best_t：{best_t:.4f}（以 MAE 最小）")
    print("----- 测试集指标 -----")
    for k, v in m_test.items():
        print(f"{k}: {v:.6f}")

    out = pd.DataFrame({
        "student_id": df.loc[test_idx, "student_id"].values if "student_id" in df.columns else np.arange(len(test_idx)),
        "true_exam": y_test,
        "pred_threshold": pred_test
    })
    out.to_csv("threshold_baseline_test_predictions.csv", index=False, encoding="utf-8-sig")
    print("已输出：threshold_baseline_test_predictions.csv")

if __name__ == "__main__":
    main()
