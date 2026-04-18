import os
import csv
import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INPUT_CSV = "exam_predictions_summary_expected_raw.csv"
OUT_DIR = "radar_charts"


def parse_mapping(cfg_str):
    raw_map = json.loads(cfg_str)
    final_map = {}
    for k, v in raw_map.items():
        k = k.strip()
        if "-" in k:
            start, end = map(int, k.split("-"))
            for i in range(start, end + 1):
                final_map[i] = v
        elif "," in k:
            for i in k.split(","):
                if i.strip():
                    final_map[int(i.strip())] = v
        else:
            final_map[int(k)] = v
    return final_map


# 🌟 完美的兜底配置：如果网页没保存配置，就用这个原版的 9 知识点映射
DEFAULT_MAPPING = {
    **{i: 1 for i in range(1, 5)},
    **{i: 2 for i in range(5, 15)},
    15: 3,
    **{i: 4 for i in [16, 17, 19, 31]},
    18: 5,
    **{i: 6 for i in [20, 21, 22, 32, 33]},
    **{i: 7 for i in [23, 24, 34]},
    **{i: 8 for i in [25, 26, 27, 29, 30, 35]},
    28: 9
}

# 读取网页配置
config_file = "config_mapping.json"
if os.path.exists(config_file):
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            EXAM_QID_TO_PID = parse_mapping(f.read())
    except:
        EXAM_QID_TO_PID = DEFAULT_MAPPING
else:
    EXAM_QID_TO_PID = DEFAULT_MAPPING

Q_COUNT_CFG = max(EXAM_QID_TO_PID.keys()) if EXAM_QID_TO_PID else 35
KP_COUNT = max(EXAM_QID_TO_PID.values()) if EXAM_QID_TO_PID else 9

KP_LABELS = [f"知识点{i}" for i in range(1, KP_COUNT + 1)]


def find_sid_col(fieldnames: List[str]) -> str:
    for c in ["student_id", "sid", "id"]:
        if c in fieldnames: return c
    return fieldnames[0]


def radar_plot(values: List[float], labels: List[str], title: str, out_path: str) -> None:
    vals = np.array(values, dtype=float)
    vals = np.clip(vals, 0.0, 1.0)

    # 如果只有一个知识点，画不了多边形，直接拦截
    if len(labels) <= 2:
        return

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals = np.concatenate([vals, vals[:1]])
    angles = angles + angles[:1]

    plt.figure(figsize=(5.2, 5.2))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=12, pad=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        dr = csv.DictReader(f)
        fieldnames = dr.fieldnames or []
        sid_col = find_sid_col(fieldnames)
        raw_rows = list(dr)

    # 容错提取
    available_prob_cols = {}
    for i in range(1, Q_COUNT_CFG + 1):
        for prefix in [f"p{i}", f"prob_q{i}", f"q{i}_prob", f"prob{i}", f"pred_q{i}"]:
            if prefix in fieldnames:
                available_prob_cols[i] = prefix
                break

    n_chart = 0
    for r in raw_rows:
        sid = r[sid_col].strip()
        kp_sum = [0.0] * KP_COUNT
        kp_cnt = [0] * KP_COUNT

        for qi, col_name in available_prob_cols.items():
            pid = EXAM_QID_TO_PID.get(qi)
            if pid is None: continue
            try:
                p = float(r[col_name])
                idx = pid - 1
                if 0 <= idx < KP_COUNT:
                    kp_sum[idx] += p
                    kp_cnt[idx] += 1
            except:
                continue

        # 如果某知识点没题，按 0 分算，绝不跳过
        kp_avg = [kp_sum[i] / kp_cnt[i] if kp_cnt[i] > 0 else 0.0 for i in range(KP_COUNT)]

        title = f"学生 {sid} 知识点掌握度"
        out_path = os.path.join(OUT_DIR, f"radar_{sid}.png")
        radar_plot(kp_avg, KP_LABELS, title, out_path)
        n_chart += 1


if __name__ == "__main__":
    main()