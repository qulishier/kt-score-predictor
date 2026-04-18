import streamlit as st
import pandas as pd
import os
import subprocess
import sys
import json

# 设置页面配置
st.set_page_config(page_title="知识追踪与成绩预测系统", layout="wide")

# ==========================================
# 核心初始化
# ==========================================
data_dir = os.path.join("data", "my_data")
os.makedirs(data_dir, exist_ok=True)

train_path = os.path.join(data_dir, "my_data_train.csv")
valid_path = os.path.join(data_dir, "my_data_valid.csv")
test_path = os.path.join(data_dir, "my_data_test.csv")
math_path = os.path.join(data_dir, "math.csv")
exam_path = os.path.join(data_dir, "exam.csv")

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# 侧边栏导航
st.sidebar.title("系统导航")
menu = st.sidebar.radio("请选择功能模块：", ["1. 算法引擎训练舱", "2. 学生综合能力画像"])

# ==========================================
# 新增：侧边栏知识点映射配置中心
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("知识点映射配置")

# 默认映射字符串
default_mapping = """{
    "1-4": 1,
    "5-14": 2,
    "15": 3,
    "16,17,19,31": 4,
    "18": 5,
    "20,21,22,32,33": 6,
    "23,24,34": 7,
    "25-27,29-30,35": 8,
    "28": 9
}"""

config_file = "config_mapping.json"
if os.path.exists(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        current_mapping = f.read()
else:
    current_mapping = default_mapping

mapping_str = st.sidebar.text_area("编辑题目与知识点的对应关系(JSON)：", value=current_mapping, height=220)

if st.sidebar.button("保存映射配置"):
    try:
        json.loads(mapping_str)
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(mapping_str)
        st.sidebar.success("✅ 配置保存成功！请去【算法引擎训练舱】重新运行流水线生效。")
    except Exception as e:
        st.sidebar.error("❌ JSON 格式有误，请检查是否漏了双引号或逗号！")

# ==========================================
# 模块一：算法引擎训练舱
# ==========================================
if menu == "1. 算法引擎训练舱":
    st.title("核心算法引擎与全栈数据流水线")


    st.markdown("### 第 1 步：上传数据集 (需 5 个文件)")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_file = st.file_uploader("上传 train.csv", type=["csv"])
    with col2:
        valid_file = st.file_uploader("上传 valid.csv", type=["csv"])
    with col3:
        test_file = st.file_uploader("上传 test.csv", type=["csv"])

    col4, col5, _ = st.columns(3)
    with col4:
        math_file = st.file_uploader("上传 math.csv", type=["csv"])
    with col5:
        exam_file = st.file_uploader("上传 exam.csv", type=["csv"])

    # 5 个文件集齐，召唤神龙
    if train_file and valid_file and test_file and math_file and exam_file:
        with open(train_path, "wb") as f: f.write(train_file.getbuffer())
        with open(valid_path, "wb") as f: f.write(valid_file.getbuffer())
        with open(test_path, "wb") as f: f.write(test_file.getbuffer())
        with open(math_path, "wb") as f: f.write(math_file.getbuffer())
        with open(exam_path, "wb") as f: f.write(exam_file.getbuffer())

        st.session_state.data_loaded = True
        st.success("✅ 5个数据文件已成功强制写入本地硬盘！系统已就绪。")

    st.markdown("###  第 2 步：启动流水线")
    if st.button("一键全自动运行流水线 (含模型训练)"):
        try:
            steps = [
                ("1/5 正在进行深度学习模型训练 (main.py)", "main.py"),
                ("2/5 正在进行特征预测与分数校准 (predict_score.py)", "predict_score.py"),
                ("3/5 正在融合成绩计算最终概率 (calc_prob.py)", "calc_prob.py"),
                ("4/5 正在处理高数特征与数据计算 (calc_math.py)", "calc_math.py"),
                ("5/5 正在根据最新配置生成雷达图 (radar_chart.py)", "radar_chart.py")
            ]
            for msg, script in steps:
                with st.spinner(msg):
                    subprocess.run([sys.executable, script], check=True)
            st.success("全链路运算完毕！请点击左侧【2. 学生综合能力画像】查看结果。")
        except Exception as e:
            st.error(f"❌ 运行报错：{e}")

# ==========================================
# 模块二：学生综合能力画像
# ==========================================
elif menu == "2. 学生综合能力画像":
    st.title("学生综合能力与期末成绩预测看板")

    ability_csv = "exam_predictions_summary_expected_raw.csv"

    if not os.path.exists(ability_csv):
        st.warning("找不到数据！请先在【算法引擎训练舱】跑通流水线。")
    else:
        df_ability = pd.read_csv(ability_csv, dtype=str)
        df_ability.columns = df_ability.columns.str.strip()

        sid_col = "student_id" if "student_id" in df_ability.columns else df_ability.columns[0]
        df_ability[sid_col] = df_ability[sid_col].str.strip()

        st.sidebar.markdown("---")
        selected_sid = st.sidebar.selectbox("搜索并选择学生学号：", sorted(df_ability[sid_col].unique()))

        stu_ability = df_ability[df_ability[sid_col] == selected_sid].iloc[0]

        st.markdown(f"### 核心预测指标 (学生: {selected_sid})")
        pred_val = stu_ability.get('raw_expected_total', stu_ability.get('pred_calibrated', '0'))

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="模型预测期末总分", value=f"{float(pred_val):.1f} 分")
        with col_m2:
            st.metric(label="预测状态", value="运算完成")

        st.markdown("---")

        st.markdown(f"### 知识点微观掌握画像 (学号: {selected_sid})")
        img_found = None
        for folder in ["radar_charts", "radar_anonym", "."]:
            test_path = os.path.join(folder, f"radar_{selected_sid}.png")
            if os.path.exists(test_path):
                img_found = test_path
                break

        if img_found:
            _, col_img, _ = st.columns([0.5, 2, 0.5])
            with col_img:
                st.image(img_found, use_column_width=True)
        else:
            st.error(f"❌ 未找到雷达图文件 (尝试路径: radar_{selected_sid}.png)")