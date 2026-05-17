# main_backtest.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 從獨立的指標計算模組中引入核心計算函式
from metrics_calc import calculate_all_metrics

# ==========================================
# 1. 讀取並預處理台積電 20 年歷史資料
# ==========================================
df = pd.read_csv('2330_TW_20y.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 初始化設定 (專案指定初始資本 USD 10,000)
initial_capital = 10000
N = 20  # MA 策略天數

print(f"【系統通知】成功載入資料！歷史資料區間：{df['Date'].min().strftime('%Y-%m-%d')} 至 {df['Date'].max().strftime('%Y-%m-%d')}")

# ==========================================
# 2. 策略一：Buy-and-Hold (B&H, 基準一)
# ==========================================
df['Market_Return'] = df['Close'].pct_change().fillna(0)
df['BH_Equity'] = initial_capital * (1 + df['Market_Return']).cumprod()

# ==========================================
# 3. 策略二：投影片黃金/死亡交叉 MA 策略
# ==========================================
df['MA'] = df['Close'].rolling(window=N).mean()
df['Price_d_1'] = df['Close'].shift(1)
df['MA_d_1'] = df['MA'].shift(1)

df['Signal'] = 0
buy_cond = (df['Price_d_1'] < df['MA_d_1']) & (df['Close'] > df['MA'])
sell_cond = (df['Price_d_1'] > df['MA_d_1']) & (df['Close'] < df['MA'])
df.loc[buy_cond, 'Signal'] = 1
df.loc[sell_cond, 'Signal'] = -1

# 計算每日持倉 (1代表持有股票，0代表空手)
df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
df['Strategy_Return'] = (df['Position'].shift(1) * df['Market_Return']).fillna(0)
df['MA_Equity'] = initial_capital * (1 + df['Strategy_Return']).cumprod()

# ==========================================
# 4. 策略三：Dollar-Cost Averaging (DCA, 定期定額, 基準二)
# ==========================================
# 設定每個月固定投入金額。假設 20 年總共投入 10,000 元
# 我們設定在每個月的第一個交易日執行扣款買入
df['YearMonth'] = df['Date'].dt.to_period('M')
df['Is_Month_First'] = df['YearMonth'] != df['YearMonth'].shift(1)

# 計算總共有幾個月需要扣款
total_months = df['Is_Month_First'].sum()
monthly_investment = initial_capital / total_months  # 每月固定扣款金額

# 模擬 DCA 滾動過程
cash = initial_capital
shares_owned = 0
dca_equity_list = []
dca_returns_list = []
prev_equity = initial_capital

for idx, row in df.iterrows():
    # 如果是每個月第一個交易日，且手頭還有現金，就扣款買入股票
    if row['Is_Month_First'] and cash >= monthly_investment:
        cash -= monthly_investment
        shares_owned += monthly_investment / row['Close']
    
    # 每日結算該策略的總資產 = 剩餘現金 + 現有股票市值
    current_equity = cash + (shares_owned * row['Close'])
    dca_equity_list.append(current_equity)
    
    # 計算 DCA 的每日報酬率 (用於算波動率)
    daily_ret = (current_equity - prev_equity) / prev_equity if idx > 0 else 0
    dca_returns_list.append(daily_ret)
    prev_equity = current_equity

df['DCA_Equity'] = dca_equity_list
df['DCA_Return'] = dca_returns_list

# ==========================================
# 5. 匯出 CSV 檔案一：每日詳細回測軌跡 (新增 DCA 淨值)
# ==========================================
df_daily_details = df[['Date', 'Close', 'MA', 'Position', 'BH_Equity', 'MA_Equity', 'DCA_Equity']].copy()
df_daily_details.to_csv('backtest_daily_details.csv', index=False)
print("👉 【已匯出】每日詳細操作與淨值紀錄 -> 'backtest_daily_details.csv'")

# ==========================================
# 6. 呼叫計算模組並匯出 CSV 檔案二：三策略績效總結表
# ==========================================
ma_metrics = calculate_all_metrics(df['MA_Equity'], df['Strategy_Return'], df['Date'])
bh_metrics = calculate_all_metrics(df['BH_Equity'], df['Market_Return'], df['Date'])
dca_metrics = calculate_all_metrics(df['DCA_Equity'], df['DCA_Return'], df['Date'])

# 建立包含三個策略的對比 DataFrame
summary_df = pd.DataFrame([bh_metrics, dca_metrics, ma_metrics], 
                          index=['Buy_and_Hold_Benchmark', 'Dollar_Cost_Averaging_Benchmark', f'MA_{N}_Crossover'])

# 轉換為百分比字串並儲存 (相容新版 Pandas 移除 applymap 的問題)
if hasattr(summary_df, 'map'):
    summary_df_formatted = summary_df.map(lambda x: f"{x:.2%}")
else:
    summary_df_formatted = summary_df.applymap(lambda x: f"{x:.2%}")

summary_df_formatted.to_csv('strategy_performance_summary.csv')
print("👉 【已匯出】三項核心測量指標總結表 -> 'strategy_performance_summary.csv'")

# ==========================================
# 7. 資料視覺化：繪製包含 DCA 的雙子圖並儲存
# ==========================================
plt.figure(figsize=(14, 10))

# 【子圖一：三策略資產增長曲線對比】
plt.subplot(2, 1, 1)
plt.plot(df['Date'], df['BH_Equity'], label=f"Buy & Hold (Final: ${df['BH_Equity'].iloc[-1]:.0f})", color='gray', alpha=0.6)
plt.plot(df['Date'], df['DCA_Equity'], label=f"DCA (Final: ${df['DCA_Equity'].iloc[-1]:.0f})", color='orange', alpha=0.8)
plt.plot(df['Date'], df['MA_Equity'], label=f"MA {N} Crossover (Final: ${df['MA_Equity'].iloc[-1]:.0f})", color='blue')
plt.title(f"2330.TW 20-Year Backtest: Three Strategies Comparison (Initial: ${initial_capital})", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 【子圖二：三策略最大回撤風險陰影圖】
df['BH_Peak'] = df['BH_Equity'].cummax()
df['BH_DD'] = (df['BH_Peak'] - df['BH_Equity']) / df['BH_Peak']
df['MA_Peak'] = df['MA_Equity'].cummax()
df['MA_DD'] = (df['MA_Peak'] - df['MA_Equity']) / df['MA_Peak']
df['DCA_Peak'] = df['DCA_Equity'].cummax()
df['DCA_DD'] = (df['DCA_Peak'] - df['DCA_Equity']) / df['DCA_Peak']

plt.subplot(2, 1, 2)
plt.fill_between(df['Date'], -df['BH_DD']*100, 0, label=f"Buy & Hold MDD: -{summary_df.loc['Buy_and_Hold_Benchmark', 'Maximum_Drawdown']:.2%}", color='gray', alpha=0.2)
plt.fill_between(df['Date'], -df['DCA_DD']*100, 0, label=f"DCA MDD: -{summary_df.loc['Dollar_Cost_Averaging_Benchmark', 'Maximum_Drawdown']:.2%}", color='orange', alpha=0.3)
plt.fill_between(df['Date'], -df['MA_DD']*100, 0, label=f"MA Crossover MDD: -{summary_df.loc[f'MA_{N}_Crossover', 'Maximum_Drawdown']:.2%}", color='blue', alpha=0.4)
plt.title("Drawdown Analysis (Risk Measure Visualized Across 3 Strategies)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_performance_chart.png', dpi=300)
plt.show()
print("👉 【已繪製】三策略績效分析對比圖已儲存 -> 'backtest_performance_chart.png'")
print("\n【回測流程全部順利完成！Problem 1 的程式與對比圖已搞定。】")