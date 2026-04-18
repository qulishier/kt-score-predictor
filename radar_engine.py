import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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
EXAM_QID_TO_PID[28] = 9
KP_COUNT = 9

# 将标签改回您原图中的 K1, K2... 格式
KP_LABELS = [f"K{i}" for i in range(1, KP_COUNT + 1)]


def get_student_mastery(row, prob_cols):
    """底层逻辑 100% 保持不变"""
    kp_sum = [0.0] * KP_COUNT
    kp_cnt = [0] * KP_COUNT

    for qi in range(1, 36):
        pid = EXAM_QID_TO_PID.get(qi)
        if pid is None: continue

        col = prob_cols[qi - 1]
        try:
            p = float(row[col])
            idx = pid - 1
            if 0 <= idx < KP_COUNT:
                kp_sum[idx] += p
                kp_cnt[idx] += 1
        except:
            continue

    kp_avg = []
    for i in range(KP_COUNT):
        if kp_cnt[i] == 0:
            kp_avg.append(float("nan"))
        else:
            kp_avg.append(kp_sum[i] / kp_cnt[i])

    return kp_avg


def plot_radar_figure(values, title):
    vals = np.array(values, dtype=float)
    vals = np.nan_to_num(vals, nan=0.0)
    vals = np.clip(vals, 0.0, 1.0)

    angles = np.linspace(0, 2 * np.pi, len(KP_LABELS), endpoint=False).tolist()
    vals = np.concatenate([vals, vals[:1]])
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = plt.subplot(111, polar=True)

    # ================= 修改核心区域：调整雷达图旋转角度 =================
    ax.set_theta_offset(np.pi / 2)  # 把 0 度位置设置在正上方 (12点钟方向)
    ax.set_theta_direction(-1)  # 设置为顺时针方向绘制
    # ====================================================================

    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(KP_LABELS, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 1.0)

    ax.set_title(title, fontsize=12, pad=20)
    plt.tight_layout()
    return fig