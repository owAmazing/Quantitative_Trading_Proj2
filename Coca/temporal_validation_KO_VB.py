# temporal_validation_volatility.py
import numpy as np
import pandas as pd
from metrics_calc import calculate_all_metrics

target_stock = 'KO' 

file_name = f'{target_stock}_20y.csv'
df = pd.read_csv(file_name)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Year'] = df['Date'].dt.year

initial_capital = 10000
candidate_n = [5, 10, 20, 60, 120]  # 布林通道的候選天數 (計算基準)
k = 2  # 標準差倍數 (固定為 2 倍標準差)

validation_results = []

print(f"🚀 【{target_stock} 系統啟動】開始執行 Volatility-adjusted (布林通道 %b) 時序驗證...")

# ==========================================
# 2. Expanding Window 滾動迴圈
# ==========================================
for train_end_year in range(2006, 2026):
    test_start_year = train_end_year + 1
    window_num = train_end_year - 2005
    
    print(f"\n⚡ [Window {window_num:02d}] 訓練: 2005~{train_end_year} | 測試: {test_start_year}~2026")
    
    train_df = df[df['Year'] <= train_end_year].copy().reset_index(drop=True)
    
    # ------------------------------------------
    # 步驟 A: 訓練期 波動率參數尋優 (尋找最佳的滾動天數 n)
    # ------------------------------------------
    best_train_return = -float('inf')
    best_n = 20  # 預設值
    
    for n in candidate_n:
        temp_train = train_df.copy()
        
        # 計算基礎布林線
        temp_train['MA'] = temp_train['Close'].rolling(window=n).mean()
        temp_train['Std'] = temp_train['Close'].rolling(window=n).std()
        temp_train['Lower_Band'] = temp_train['MA'] - (k * temp_train['Std'])
        temp_train['Upper_Band'] = temp_train['MA'] + (k * temp_train['Std'])
        
        # 引入 %b 指標
        temp_train['Percent_B'] = (temp_train['Close'] - temp_train['Lower_Band']) / (temp_train['Upper_Band'] - temp_train['Lower_Band'])
        
        # 使用 %b 建立訊號
        temp_train['Signal'] = 0
        buy_cond = (temp_train['Percent_B'].shift(1) >= 0) & (temp_train['Percent_B'] < 0)
        sell_cond = (temp_train['Percent_B'].shift(1) <= 1) & (temp_train['Percent_B'] > 1)
        
        temp_train.loc[buy_cond, 'Signal'] = 1
        temp_train.loc[sell_cond, 'Signal'] = -1
        
        # 持倉滾動 (向量化連續技)
        temp_train['Position'] = temp_train['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
        
        # 計算資產曲線
        mkt_ret = temp_train['Close'].pct_change().fillna(0)
        strat_ret = (temp_train['Position'].shift(1) * mkt_ret).fillna(0)
        equity_curve = initial_capital * (1 + strat_ret).cumprod()
        
        # 尋找累積報酬率最高的 n
        final_train_ret = (equity_curve.iloc[-1] / initial_capital) - 1
        if final_train_ret > best_train_return:
            best_train_return = final_train_ret
            best_n = n
            
    print(f"   👉 訓練完成！最佳參數為 BB n = {best_n:3d} (訓練期報酬: {best_train_return:.2%})")
    
    # ------------------------------------------
    # 步驟 B: 抱著最佳參數，進入未知「測試期」盲測
    # ------------------------------------------
    df_global = df.copy()
    # 依據選出的 best_n 計算全局布林通道
    df_global['MA'] = df_global['Close'].rolling(window=best_n).mean()
    df_global['Std'] = df_global['Close'].rolling(window=best_n).std()
    df_global['Lower_Band'] = df_global['MA'] - (k * df_global['Std'])
    df_global['Upper_Band'] = df_global['MA'] + (k * df_global['Std'])
    
    # 【修正：步驟 B 同步升級 %b 指標，確保前後邏輯一致】
    df_global['Percent_B'] = (df_global['Close'] - df_global['Lower_Band']) / (df_global['Upper_Band'] - df_global['Lower_Band'])
    
    df_global['Signal'] = 0
    buy_cond = (df_global['Percent_B'].shift(1) >= 0) & (df_global['Percent_B'] < 0)
    sell_cond = (df_global['Percent_B'].shift(1) <= 1) & (df_global['Percent_B'] > 1)
    df_global.loc[buy_cond, 'Signal'] = 1
    df_global.loc[sell_cond, 'Signal'] = -1
    df_global['Position'] = df_global['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
    
    # 擷取測試期資料
    test_df = df_global[(df_global['Year'] >= test_start_year) & (df_global['Year'] <= 2026)].copy().reset_index(drop=True)
    
    # 計算測試期表現
    test_df['Market_Return'] = test_df['Close'].pct_change().fillna(0)
    test_df['Strategy_Return'] = (test_df['Position'].shift(1) * test_df['Market_Return']).fillna(0)
    test_df['BB_Equity'] = initial_capital * (1 + test_df['Strategy_Return']).cumprod()
    
    # 呼叫核心計算機 (metrics_calc.py)
    test_metrics = calculate_all_metrics(test_df['BB_Equity'], test_df['Strategy_Return'], test_df['Date'])
    
    # 紀錄 Window 成果
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
# 3. 彙整結果輸出 CSV 報告
# ==========================================
temporal_summary_df = pd.DataFrame(validation_results)

# 格式化百分比字串
formatted_df = temporal_summary_df.copy()
percent_cols = ["Test_Cum_Return", "Test_Ann_Return", "Test_Volatility"]
for col in percent_cols:
    formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2%}")
formatted_df["Test_MDD"] = formatted_df["Test_MDD"].map(lambda x: f"-{x:.2%}")

output_file = f'{target_stock}_volatility_validation_%b_summary.csv'
formatted_df.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"🎉 【大功告成】{target_stock} 波動率調節策略時序驗證全部跑完！")
print(f"👉 成果報告已成功儲存至 -> '{output_file}'")
print("="*60)
print(formatted_df.to_string(index=False))