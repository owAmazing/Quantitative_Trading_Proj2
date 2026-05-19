# temporal_validation.py
import numpy as np
import pandas as pd
# 引入核心計算機 (請確保 metrics_calc.py 在同一個資料夾)
from metrics_calc import calculate_all_metrics

# ==========================================
# 1. 讀取並預處理台積電資料
# ==========================================
df = pd.read_csv('AAPL_20y.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 建立年份欄位方便切分時間軸
df['Year'] = df['Date'].dt.year

initial_capital = 10000
candidate_N = [5, 10, 20, 60, 120]  # 網格搜尋的均線候選參數

# 用來儲存每一個 Window 測試期成果的清單
validation_results = []

print("🚀 【系統啟動】開始執行完整 Expanding Window 時序驗證...")
print("   時間範圍：從 (Train: 2005-2006 / Test: 2007-2026) 滾動至 (Train: 2005-2025 / Test: 2026)")

# ==========================================
# 2. Expanding Window 核心滾動迴圈
# ==========================================
# 訓練結束年份從 2006 年一路滾動增長到 2025 年
for train_end_year in range(2006, 2026):
    test_start_year = train_end_year + 1
    window_num = train_end_year - 2005
    
    print(f"\n⚡ [Window {window_num}] 訓練區間: 2005~{train_end_year} | 測試區間: {test_start_year}~2026")
    
    # 切出當前 Window 的訓練集 (2005 到 train_end_year)
    train_df = df[df['Year'] <= train_end_year].copy().reset_index(drop=True)
    
    # ------------------------------------------
    # 步驟 A: 在該 Window 的訓練期進行「參數尋優 (Optimization)」
    # ------------------------------------------
    best_train_return = -float('inf')
    best_N = 20  # 預設值
    
    for N in candidate_N:
        temp_train = train_df.copy()
        temp_train['MA'] = temp_train['Close'].rolling(window=N).mean()
        
        # 建立訊號 (d-1 昨天與今天比對)
        temp_train['Signal'] = 0
        buy_cond = (temp_train['Close'].shift(1) < temp_train['MA'].shift(1)) & (temp_train['Close'] > temp_train['MA'])
        sell_cond = (temp_train['Close'].shift(1) > temp_train['MA'].shift(1)) & (temp_train['Close'] < temp_train['MA'])
        temp_train.loc[buy_cond, 'Signal'] = 1
        temp_train.loc[sell_cond, 'Signal'] = -1
        
        # 持倉狀態滾動
        temp_train['Position'] = temp_train['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
        
        # 計算訓練期資產曲線
        mkt_ret = temp_train['Close'].pct_change().fillna(0)
        strat_ret = (temp_train['Position'].shift(1) * mkt_ret).fillna(0)
        equity_curve = initial_capital * (1 + strat_ret).cumprod()
        
        # 紀錄訓練期累積報酬率最高的 N
        final_train_ret = (equity_curve.iloc[-1] / initial_capital) - 1
        if final_train_ret > best_train_return:
            best_train_return = final_train_ret
            best_N = N
            
    print(f"   👉 訓練完成！最佳參數為 N = {best_N:3d} (該區間訓練報酬率: {best_train_return:.2%})")
    
    # ------------------------------------------
    # 步驟 B: 抱著最佳參數 best_N，去未知的「測試期」進行盲測
    # ------------------------------------------
    # 在全局數據計算 MA，確保測試期接頭處有完整的歷史數據算 MA，避免產生 NaN 誤差
    df_global = df.copy()
    df_global['MA'] = df_global['Close'].rolling(window=best_N).mean()
    
    df_global['Signal'] = 0
    buy_cond = (df_global['Close'].shift(1) < df_global['MA'].shift(1)) & (df_global['Close'] > df_global['MA'])
    sell_cond = (df_global['Close'].shift(1) > df_global['MA'].shift(1)) & (df_global['Close'] < df_global['MA'])
    df_global.loc[buy_cond, 'Signal'] = 1
    df_global.loc[sell_cond, 'Signal'] = -1
    df_global['Position'] = df_global['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
    
    # 切出屬於該 Window 的獨立測試集區間 (從 test_start_year 到最後一年 2026)
    test_df = df_global[(df_global['Year'] >= test_start_year) & (df_global['Year'] <= 2026)].copy().reset_index(drop=True)
    
    # 計算測試期的每日報酬率與資產淨值曲線
    test_df['Market_Return'] = test_df['Close'].pct_change().fillna(0)
    test_df['Strategy_Return'] = (test_df['Position'].shift(1) * test_df['Market_Return']).fillna(0)
    test_df['MA_Equity'] = initial_capital * (1 + test_df['Strategy_Return']).cumprod()
    
    # 呼跨核心計算機算核心指標
    test_metrics = calculate_all_metrics(test_df['MA_Equity'], test_df['Strategy_Return'], test_df['Date'])
    
    # 紀錄該 Window 的最終盲測成果
    validation_results.append({
        "Window": f"Window_{window_num:02d}",
        "Train_Period": f"2005-{train_end_year}",
        "Test_Period": f"{test_start_year}-2026",
        "Selected_Best_N": best_N,
        "Test_Cum_Return": test_metrics['Cumulative_Return'],
        "Test_Ann_Return": test_metrics['Annualized_Return'],
        "Test_MDD": test_metrics['Maximum_Drawdown'],
        "Test_Volatility": test_metrics['Volatility']
    })

# ==========================================
# 3. 彙整所有 Window 成果並輸出 CSV 報告
# ==========================================
temporal_summary_df = pd.DataFrame(validation_results)

# 將數值格式化為百分比字串方便閱讀
formatted_df = temporal_summary_df.copy()
percent_cols = ["Test_Cum_Return", "Test_Ann_Return", "Test_Volatility"]
for col in percent_cols:
    formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2%}")
formatted_df["Test_MDD"] = formatted_df["Test_MDD"].map(lambda x: f"-{x:.2%}")

# 匯出成 CSV
formatted_df.to_csv('temporal_validation_summary_AAPL_SMA.csv', index=False)
print("\n" + "="*50)
print("🎉 【大功告成】完整滾動時序驗證已全部執行完畢！")
print("👉 成果報告已成功儲存至 -> 'temporal_validation_summary_AAPL_SMA.csv'")
print("="*50)

# 在終端機印出前幾行與最後幾行給你看
print(formatted_df.to_string(index=False))