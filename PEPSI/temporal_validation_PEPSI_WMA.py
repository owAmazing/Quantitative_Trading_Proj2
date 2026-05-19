# temporal_validation_wma.py
import numpy as np
import pandas as pd
from metrics_calc import calculate_all_metrics

# ==========================================
# 1. 讀取並預處理 Pepsi (PEP) 資料
# ==========================================
df = pd.read_csv('PEP_20y.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Year'] = df['Date'].dt.year

initial_capital = 10000
candidate_n = [5, 10, 20, 60, 120]  # WMA 的候選週期數 n

validation_results = []

print("🚀 【AAPL 蘋果公司系統啟動】開始執行 WMA 擴展窗口時序驗證...")

# ==========================================
# 2. 定義計算 WMA 的自訂函數 (符合使用者公式)
# ==========================================
def calculate_wma_series(series, n):
    """
    自訂 WMA 計算機
    Pandas rolling.apply 傳入的陣列順序是 [最舊的價格, ..., 最新的價格]
    根據公式：最新的價格 (最後一個元素) 權重要最大 (n)，最舊的 (第一個元素) 權重要最小 (1)
    """
    weights = np.arange(1, n + 1) # 產生 [1, 2, ..., n] 的權重陣列
    
    # 使用 rolling.apply 進行滾動計算
    return series.rolling(window=n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# ==========================================
# 3. Expanding Window 滾動迴圈
# ==========================================
for train_end_year in range(2006, 2026):
    test_start_year = train_end_year + 1
    window_num = train_end_year - 2005
    
    print(f"\n⚡ [Window {window_num:02d}] 訓練: 2005~{train_end_year} | 測試: {test_start_year}~2026")
    
    # 切出訓練集
    train_df = df[df['Year'] <= train_end_year].copy().reset_index(drop=True)
    
    # ------------------------------------------
    # 步驟 A: 訓練期 WMA 參數尋優
    # ------------------------------------------
    best_train_return = -float('inf')
    best_n = 20  # 預設值
    
    for n in candidate_n:
        temp_train = train_df.copy()
        
        # 計算 WMA 欄位
        temp_train['WMA'] = calculate_wma_series(temp_train['Close'], n)
        
        # 建立 WMA 策略訊號 (昨日 Close 與昨日 WMA 比較，今日觸發)
        temp_train['Signal'] = 0
        buy_cond = (temp_train['Close'].shift(1) < temp_train['WMA'].shift(1)) & (temp_train['Close'] > temp_train['WMA'])
        sell_cond = (temp_train['Close'].shift(1) > temp_train['WMA'].shift(1)) & (temp_train['Close'] < temp_train['WMA'])
        temp_train.loc[buy_cond, 'Signal'] = 1
        temp_train.loc[sell_cond, 'Signal'] = -1
        
        # 持倉狀態滾動 (前向填充向量化連續技)
        temp_train['Position'] = temp_train['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
        
        # 計算訓練期資產曲線
        mkt_ret = temp_train['Close'].pct_change().fillna(0)
        strat_ret = (temp_train['Position'].shift(1) * mkt_ret).fillna(0)
        equity_curve = initial_capital * (1 + strat_ret).cumprod()
        
        # 以累積報酬率挑選該區間最佳的 n
        final_train_ret = (equity_curve.iloc[-1] / initial_capital) - 1
        if final_train_ret > best_train_return:
            best_train_return = final_train_ret
            best_n = n
            
    print(f"   👉 訓練完成！最佳參數為 WMA n = {best_n:3d} (訓練期報酬: {best_train_return:.2%})")
    
    # ------------------------------------------
    # 步驟 B: 抱著最佳 WMA 參數，進入未知「測試期」盲測
    # ------------------------------------------
    df_global = df.copy()
    # 依據選出的 best_n 計算全局 WMA
    df_global['WMA'] = calculate_wma_series(df_global['Close'], best_n)
    
    df_global['Signal'] = 0
    buy_cond = (df_global['Close'].shift(1) < df_global['WMA'].shift(1)) & (df_global['Close'] > df_global['WMA'])
    sell_cond = (df_global['Close'].shift(1) > df_global['WMA'].shift(1)) & (df_global['Close'] < df_global['WMA'])
    df_global.loc[buy_cond, 'Signal'] = 1
    df_global.loc[sell_cond, 'Signal'] = -1
    df_global['Position'] = df_global['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
    
    # 擷取測試期資料 (test_start_year ~ 2026)
    test_df = df_global[(df_global['Year'] >= test_start_year) & (df_global['Year'] <= 2026)].copy().reset_index(drop=True)
    
    # 計算測試期資產表現
    test_df['Market_Return'] = test_df['Close'].pct_change().fillna(0)
    test_df['Strategy_Return'] = (test_df['Position'].shift(1) * test_df['Market_Return']).fillna(0)
    test_df['WMA_Equity'] = initial_capital * (1 + test_df['Strategy_Return']).cumprod()
    
    # 呼叫核心計算機
    test_metrics = calculate_all_metrics(test_df['WMA_Equity'], test_df['Strategy_Return'], test_df['Date'])
    
    # 紀錄該 Window 成果
    validation_results.append({
        "Window": f"Window_{window_num:02d}",
        "Train_Period": f"2005-{train_end_year}",
        "Test_Period": f"{test_start_year}-2026",
        "Selected_Best_n": best_n,
        "Test_Cum_Return": test_metrics['Cumulative_Return'],
        "Test_Ann_Return": test_metrics['Annualized_Return'],
        "Test_MDD": test_metrics['Maximum_Drawdown'],
        "Test_Volatility": test_metrics['Volatility']
    })

# ==========================================
# 4. 彙整結果輸出 CSV 報告
# ==========================================
temporal_summary_df = pd.DataFrame(validation_results)

# 格式化百分比字串方便閱讀
formatted_df = temporal_summary_df.copy()
percent_cols = ["Test_Cum_Return", "Test_Ann_Return", "Test_Volatility"]
for col in percent_cols:
    formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2%}")
formatted_df["Test_MDD"] = formatted_df["Test_MDD"].map(lambda x: f"-{x:.2%}")

formatted_df.to_csv('PEP_wma_temporal_validation_summary.csv', index=False)
print("\n" + "="*60)
print("🎉 【大功告成】PEP WMA 時序驗證全部跑完！")
print("👉 成果報告已成功儲存至 -> 'PEP_wma_temporal_validation_summary.csv'")
print("="*60)
print(formatted_df.to_string(index=False))