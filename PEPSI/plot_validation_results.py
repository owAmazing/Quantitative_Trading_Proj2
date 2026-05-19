# plot_all_metrics_grid.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# 💡 設定區：要觀察哪一檔股票？ (可自由切換 'PEP' 或 'KO')
# =========================================================================
TARGET_STOCK = 'PEP'

# 設定圖片風格，讓報告與 PPT 更有質感
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']  # 支援中文顯示
plt.rcParams['axes.unicode_minus'] = False 

print(f"📊 正在讀取 {TARGET_STOCK} 的所有策略數據，並準備生成四大指標全方位對比圖...")

# =========================================================================
# 1. 自動尋找並讀取該股票的三個策略 CSV 檔案
# =========================================================================
files_to_load = {
    "SMA 策略": f"temporal_validation_summary_{'PEPSI' if TARGET_STOCK=='PEP' else TARGET_STOCK}_SMA.csv",
    "WMA 策略": f"{TARGET_STOCK}_wma_temporal_validation_summary.csv",
    "布林通道 %b 策略": f"{TARGET_STOCK}_volatility_validation_%b_summary.csv"
}

dfs = {}
for strategy_name, file_path in files_to_load.items():
    if os.path.exists(file_path):
        df_temp = pd.read_csv(file_path)
        
        # 數據清洗：移除百分比符號，並將字串轉換成浮點數以便計算與繪圖
        df_temp['Test_Cum_Return_Num'] = df_temp['Test_Cum_Return'].str.rstrip('%').astype('float')
        df_temp['Test_Ann_Return_Num'] = df_temp['Test_Ann_Return'].str.rstrip('%').astype('float')
        df_temp['Test_MDD_Num'] = df_temp['Test_MDD'].str.rstrip('%').astype('float')
        df_temp['Test_Volatility_Num'] = df_temp['Test_Volatility'].str.rstrip('%').astype('float')
        
        dfs[strategy_name] = df_temp
        print(f"✓ 成功載入 {strategy_name} 數據")
    else:
        print(f"⚠ 找不到檔案: {file_path}")

if not dfs:
    print("❌ 錯誤：沒有載入任何有效的 CSV 資料，請檢查檔案路徑！")
    exit()

# =========================================================================
# 2. 初始化 2x2 四宮格畫布
# =========================================================================
#  修正後的正確寫法
fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
((ax_cum, ax_ann), (ax_mdd, ax_vol)) = axes  # 正確將 2x2 矩陣解構分配給四個子圖

# 取得 X 軸窗口名稱與對應的測試期年份
sample_df = list(dfs.values())[0]
windows = sample_df['Window']
test_periods = sample_df['Test_Period']
x_labels = [f"{w}\n({p})" for w, p in zip(windows, test_periods)]

# 定義各策略的視覺外觀
styles = {
    "SMA 策略": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "WMA 策略": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "布林通道 %b 策略": {"color": "#2ca02c", "marker": "D", "linestyle": "-."}
}

# =========================================================================
# 3. 依序繪製四個指標的折線圖
# =========================================================================

# -------------------------------------------------------------------------
# (1) 左上：測試期累積報酬率 (Test Cumulative Return)
# -------------------------------------------------------------------------
for strategy_name, df_data in dfs.items():
    ax_cum.plot(windows, df_data['Test_Cum_Return_Num'], label=strategy_name,
                color=styles[strategy_name]["color"], marker=styles[strategy_name]["marker"],
                linestyle=styles[strategy_name]["linestyle"], linewidth=2)
ax_cum.set_title("(A) 測試期累積報酬率 (Test Cumulative Return)", fontsize=13, fontweight='bold', pad=10)
ax_cum.set_ylabel("報酬率 (%)", fontsize=11)
ax_cum.axhline(0, color='black', linestyle=':', alpha=0.3)
ax_cum.legend(loc="upper left", fontsize=10)

# -------------------------------------------------------------------------
# (2) 右上：測試期年化報酬率 (Test Annualized Return)
# -------------------------------------------------------------------------
for strategy_name, df_data in dfs.items():
    ax_ann.plot(windows, df_data['Test_Ann_Return_Num'], label=strategy_name,
                color=styles[strategy_name]["color"], marker=styles[styles[strategy_name]["color"]==styles[strategy_name]["color"] and strategy_name]["marker"],
                linestyle=styles[strategy_name]["linestyle"], linewidth=2)
ax_ann.set_title("(B) 測試期年化報酬率 (Test Annualized Return)", fontsize=13, fontweight='bold', pad=10)
ax_ann.set_ylabel("年化報酬率 (%)", fontsize=11)
ax_ann.axhline(0, color='black', linestyle=':', alpha=0.3)

# -------------------------------------------------------------------------
# (3) 左下：測試期最大回撤 (Test Maximum Drawdown)
# -------------------------------------------------------------------------
for strategy_name, df_data in dfs.items():
    ax_mdd.plot(windows, df_data['Test_MDD_Num'], label=strategy_name,
                color=styles[strategy_name]["color"], marker=styles[strategy_name]["marker"],
                linestyle=styles[strategy_name]["linestyle"], linewidth=2)
ax_mdd.set_title("(C) 測試期最大回撤 (Test Maximum Drawdown)", fontsize=13, fontweight='bold', pad=10)
ax_mdd.set_ylabel("MDD (%)", fontsize=11)
ax_mdd.axhline(0, color='black', linestyle=':', alpha=0.3)

# -------------------------------------------------------------------------
# (4) 右下：測試期波動度 (Test Volatility)
# -------------------------------------------------------------------------
for strategy_name, df_data in dfs.items():
    ax_vol.plot(windows, df_data['Test_Volatility_Num'], label=strategy_name,
                color=styles[strategy_name]["color"], marker=styles[strategy_name]["marker"],
                linestyle=styles[strategy_name]["linestyle"], linewidth=2)
ax_vol.set_title("(D) 測試期波動度 (Test Volatility)", fontsize=13, fontweight='bold', pad=10)
ax_vol.set_ylabel("波動度 (%)", fontsize=11)

# =========================================================================
# 4. 全局細節優化 (X 軸調整與排版)
# =========================================================================
# 為下方的兩個子圖（左下、右下）設定 X 軸標籤與旋轉角度
for ax in [ax_mdd, ax_vol]:
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel("滾動窗口代號 (對應未知測試期)", fontsize=11, fontweight='bold')

# 設定整張大圖的總標題
fig.suptitle(f"【{TARGET_STOCK}】", 
             fontsize=18, fontweight='bold', y=0.98)

# 自動調配子圖間距，防止文字和標籤跟 X 軸重疊
plt.tight_layout()

# 儲存圖片
output_image_name = f"{TARGET_STOCK}_four_metrics_grid_chart.png"
plt.savefig(output_image_name, dpi=300)

print("\n" + "="*60)
print(f"🎉 【四宮格圖表繪製成功！】")
print(f"👉 高解析度分析圖表已儲存至 -> '{output_image_name}'")
print("="*60)
plt.show()