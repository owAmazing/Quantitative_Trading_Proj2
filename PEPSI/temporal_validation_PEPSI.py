# temporal_validation_bandwidth.py
import numpy as np
import pandas as pd
from metrics_calc import calculate_all_metrics

target_stock = 'PEP'  # 可自由切換 'KO' 或 'PEP'

file_name = f'{target_stock}_20y.csv'
df = pd.read_csv(file_name)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Year'] = df['Date'].dt.year

initial_capital = 10000
candidate_n = [5, 10, 20, 30, 40, 50, 60]  # 布林通道的候選天數
k = 2 

validation_results = []

print(f"🚀 【{target_stock} 系統啟動】開始執行 Volatility-adjusted (帶寬指標 BBW 順勢突破) 時序驗證...")

# ==========================================
# 2. Expanding Window 滾動迴圈
# ==========================================
for train_end_year in range(2006, 2026):
    test_start_year = train_end_year + 1
    window_num = train_end_year - 2005
    
    print(f"\n⚡ [Window {window_num:02d}] 訓練: 2005~{train_end_year} | 測試: {test_start_year}~2026")
    
    train_df = df[df['Year'] <= train_end_year].copy().reset_index(drop=True)
    
    # ------------------------------------------
    # 步驟 A: 訓練期 帶寬參數尋優
    # ------------------------------------------
    best_train_return = -float('inf')
    best_n = 20 
    
    for n in candidate_n:
        temp_train = train_df.copy()
        
        # 計算布林線
        temp_train['MA'] = temp_train['Close'].rolling(window=n).mean()
        temp_train['Std'] = temp_train['Close'].rolling(window=n).std()
        temp_train['Lower_Band'] = temp_train['MA'] - (k * temp_train['Std'])
        temp_train['Upper_Band'] = temp_train['MA'] + (k * temp_train['Std'])
        
        # 【新功能】計算帶寬指標 (Bandwidth)
        temp_train['BBW'] = (temp_train['Upper_Band'] - temp_train['Lower_Band']) / temp_train['MA']
        # 計算帶寬的 5 日平均，用來判斷帶寬是否「放大」
        temp_train['BBW_MA5'] = temp_train['BBW'].rolling(window=5).mean()
        
        # 【新功能】建立帶寬順勢突破訊號
        temp_train['Signal'] = 0
        
        # 買入條件：今天帶寬大於 5日平均 (通道打開) + 股價突破上軌 (短線看漲追高)
        buy_cond = (temp_train['BBW'] > temp_train['BBW_MA5']) & (temp_train['Close'] > temp_train['Upper_Band'])
        
        # 賣出條件：當通道開始縮小 (BBW 跌破昨日 BBW) 或者 股價跌破中線 (動能消失)
        sell_cond = (temp_train['BBW'] < temp_train['BBW'].shift(1)) | (temp_train['Close'] < temp_train['MA'])
        
        temp_train.loc[buy_cond, 'Signal'] = 1
        temp_train.loc[sell_cond, 'Signal'] = -1
        
        # 持倉滾動
        temp_train['Position'] = temp_train['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
        
        # 計算資產曲線
        mkt_ret = temp_train['Close'].pct_change().fillna(0)
        strat_ret = (temp_train['Position'].shift(1) * mkt_ret).fillna(0)
        equity_curve = initial_capital * (1 + strat_ret).cumprod()
        
        final_train_ret = (equity_curve.iloc[-1] / initial_capital) - 1
        if final_train_ret > best_train_return:
            best_train_return = final_train_ret
            best_n = n
            
    print(f"   👉 訓練完成！最佳參數為 BBW n = {best_n:3d} (訓練期報酬: {best_train_return:.2%})")
    
    # ------------------------------------------
    # 步驟 B: 抱著最佳參數，進入未知「測試期」盲測
    # ------------------------------------------
    df_global = df.copy()
    df_global['MA'] = df_global['Close'].rolling(window=best_n).mean()
    df_global['Std'] = df_global['Close'].rolling(window=best_n).std()
    df_global['Lower_Band'] = df_global['MA'] - (k * df_global['Std'])
    df_global['Upper_Band'] = df_global['MA'] + (k * df_global['Std'])
    
    # 全局同步計算帶寬
    df_global['BBW'] = (df_global['Upper_Band'] - df_global['Lower_Band']) / df_global['MA']
    df_global['BBW_MA5'] = df_global['BBW'].rolling(window=5).mean()
    
    df_global['Signal'] = 0
    buy_cond = (df_global['BBW'] > df_global['BBW_MA5']) & (df_global['Close'] > df_global['Upper_Band'])
    sell_cond = (df_global['BBW'] < df_global['BBW'].shift(1)) | (df_global['Close'] < df_global['MA'])
    df_global.loc[buy_cond, 'Signal'] = 1
    df_global.loc[sell_cond, 'Signal'] = -1
    df_global['Position'] = df_global['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
    
    # 擷取測試期
    test_df = df_global[(df_global['Year'] >= test_start_year) & (df_global['Year'] <= 2026)].copy().reset_index(drop=True)
    
    # 計算測試期表現
    test_df['Market_Return'] = test_df['Close'].pct_change().fillna(0)
    test_df['Strategy_Return'] = (test_df['Position'].shift(1) * test_df['Market_Return']).fillna(0)
    test_df['BBW_Equity'] = initial_capital * (1 + test_df['Strategy_Return']).cumprod()
    
    test_metrics = calculate_all_metrics(test_df['BBW_Equity'], test_df['Strategy_Return'], test_df['Date'])
    
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
formatted_df = temporal_summary_df.copy()
percent_cols = ["Test_Cum_Return", "Test_Ann_Return", "Test_Volatility"]
for col in percent_cols:
    formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2%}")
formatted_df["Test_MDD"] = formatted_df["Test_MDD"].map(lambda x: f"-{x:.2%}")

output_file = f'{target_stock}_bandwidth_validation_summary.csv'
formatted_df.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"🎉 【大功告成】{target_stock} 帶寬指標策略時序驗證全部跑完！")
print(f"👉 成果報告已成功儲存至 -> '{output_file}'")
print("="*60)
print(formatted_df.to_string(index=False))