import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# 讀取 TSMC 與 MTC 的歷史收盤價資料，並對齊共同交易日
# 返回單一 DataFrame，索引為交易日期，包含 TSMC 與 MTC 收盤價
def load_data():
    tsmc = pd.read_csv('2330_TW_20y.csv', parse_dates=['Date'])
    mtc = pd.read_csv('2454_TW_20y.csv', parse_dates=['Date'])
    tsmc = tsmc[['Date', 'Close']].rename(columns={'Close': 'TSMC'})
    mtc = mtc[['Date', 'Close']].rename(columns={'Close': 'MTC'})
    df = pd.merge(tsmc, mtc, on='Date', how='inner').sort_values('Date').set_index('Date')
    df = df.dropna()
    return df


def estimate_hedge_ratio(prices_y, prices_x):
    x = prices_x.values.reshape(-1, 1).astype(float)
    y = prices_y.values.astype(float)
    model = sm.OLS(y, x).fit()
    return float(model.params[0])


def compute_spread(prices_y, prices_x, beta):
    return prices_y - beta * prices_x


def run_pair_trading(prices_a, prices_b, window=20, entry_z=2.5, exit_z=0.2, notional=10_000, beta=None):
    # 建立 A/B 兩支股票價格資料表
    df = pd.DataFrame({'A': prices_a, 'B': prices_b})

    # 計算 hedge ratio beta，如果未給定就用全資料 OLS 計算
    if beta is None:
        beta = estimate_hedge_ratio(df['A'], df['B'])
    df['beta'] = beta

    # 依照訓練期 beta 計算價差：TSMC - beta * MTC
    df['spread'] = compute_spread(df['A'], df['B'], beta)

    # 使用滾動視窗計算 spread 的平均與標準差
    df['spread_mean'] = df['spread'].rolling(window).mean()
    df['spread_std'] = df['spread'].rolling(window).std()

    # z-score = (spread - mean) / std，用於進出場訊號
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    df['zscore'] = df['zscore'].fillna(0)

    # 建立交易部位：0 = 空倉, 1 = long A short B, -1 = short A long B
    position = 0
    positions = []
    for z in df['zscore']:
        if position == 0:
            if z > entry_z:
                position = -1
            elif z < -entry_z:
                position = 1
        elif abs(z) < exit_z:
            position = 0
        positions.append(position)

    df['position'] = positions
    df['a_shares'] = 0.0
    df['b_shares'] = 0.0

    # 根據當前部位計算每檔股票的股數，假設每次進場以 notional 的一半槓桿
    current_a = 0.0
    current_b = 0.0
    for idx, row in df.iterrows():
        if row['position'] == 1:
            if current_a == 0 and current_b == 0:
                current_a = (notional / 2) / row['A']
                current_b = -(notional / 2) / row['B']
        elif row['position'] == -1:
            if current_a == 0 and current_b == 0:
                current_a = -(notional / 2) / row['A']
                current_b = (notional / 2) / row['B']
        else:
            current_a = 0.0
            current_b = 0.0

        df.at[idx, 'a_shares'] = current_a
        df.at[idx, 'b_shares'] = current_b

    # 計算部位價值與策略每日報酬
    df['position_value'] = df['a_shares'] * df['A'] + df['b_shares'] * df['B']
    df['strategy_returns'] = df['position_value'].diff().fillna(0) / notional
    df['strategy_value'] = (1 + df['strategy_returns']).cumprod() * notional
    return df.dropna()


# 計算策略績效指標，輸入為每日報酬序列
# 回傳年化報酬、年化波動、Sharpe ratio、累積報酬、最大回撤與最後資產值
def performance_metrics(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            'Annualized Return': np.nan,
            'Annualized Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Cumulative Return': np.nan,
            'Max Drawdown': np.nan,
            'Final Wealth': np.nan,
        }
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    cum_ret = (1 + returns).prod() - 1
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    return {
        'Annualized Return': ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Cumulative Return': cum_ret,
        'Max Drawdown': max_dd,
        'Final Wealth': wealth.iloc[-1],
    }


def performance_metrics_monthly(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            'Annualized Return': np.nan,
            'Annualized Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Cumulative Return': np.nan,
            'Max Drawdown': np.nan,
            'Final Wealth': np.nan,
        }
    ann_ret = (1 + returns).prod() ** (12 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    cum_ret = (1 + returns).prod() - 1
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    return {
        'Annualized Return': ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Cumulative Return': cum_ret,
        'Max Drawdown': max_dd,
        'Final Wealth': wealth.iloc[-1],
    }

# 定期定額 (DCA) 策略：每月月底分批平均買進兩檔股票
# 這裡將每月 10,000 USD 均分成 TSMC 與 MTC
def dca_strategy(prices_a, prices_b, monthly_contribution=10_000):
    df = pd.DataFrame({'A': prices_a, 'B': prices_b})
    month_ends = df.resample('M').last().index
    a_shares = 0.0
    b_shares = 0.0
    value = []
    for date, row in df.iterrows():
        if date in month_ends:
            a_shares += monthly_contribution / 2 / row['A']
            b_shares += monthly_contribution / 2 / row['B']
        value.append(a_shares * row['A'] + b_shares * row['B'])
    return pd.Series(value, index=df.index)


# 參數網格搜尋：在訓練資料上選出表現最好的 entry/exit z-score 與 rolling window 組合
# 這個函數會回傳最佳參數組合，評估指標為年化報酬
# 訓練期內只用相同參數搜尋最佳值，測試期保持參數不變
def find_best_params(prices_a, prices_b, beta):
    best_score = -np.inf
    best_params = {'window': 60, 'entry_z': 2, 'exit_z': 0.2}
    param_candidates = [
        {'window': 20, 'entry_z': 1.5, 'exit_z': 0.2},
        {'window': 20, 'entry_z': 2, 'exit_z': 0.2},
        {'window': 20, 'entry_z': 2.5, 'exit_z': 0.2},
        {'window': 40, 'entry_z': 1.5, 'exit_z': 0.2},
        {'window': 40, 'entry_z': 2, 'exit_z': 0.2},
        {'window': 40, 'entry_z': 2.5, 'exit_z': 0.2},
        {'window': 60, 'entry_z': 1.5, 'exit_z': 0.2},
        {'window': 60, 'entry_z': 2, 'exit_z': 0.2},
        {'window': 60, 'entry_z': 2.5, 'exit_z': 0.2},
    ]
    for params in param_candidates:
        result = run_pair_trading(prices_a, prices_b, beta=beta, **params)
        metrics = performance_metrics(result['strategy_returns'])
        score = metrics['Annualized Return'] if not np.isnan(metrics['Annualized Return']) else -np.inf
        if score > best_score:
            best_score = score
            best_params = params
    return best_params


# 時序驗證：使用擴展訓練期 + 未見測試期的驗證方法
# 先用 Year 1 訓練並測試 Year 2...n，再用 Years 1-2 訓練並測試 Years 3...n，以此類推
# 重要的是：beta 必須在訓練期內估計，測試期則直接套用該 beta 計算 Spread 與 Z-score
def rolling_temporal_validation(df):
    years = sorted(df.index.year.unique())
    validation_results = []

    for i in range(1, len(years)):
        train_years = years[:i]
        test_years = years[i:]
        train_mask = df.index.year.isin(train_years)
        test_mask = df.index.year.isin(test_years)
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]

        if len(train_df) < 20 or len(test_df) < 20:
            continue

        beta = estimate_hedge_ratio(train_df['TSMC'], train_df['MTC'])
        best_params = find_best_params(train_df['TSMC'], train_df['MTC'], beta)

        train_result = run_pair_trading(train_df['TSMC'], train_df['MTC'], beta=beta, **best_params)
        train_metrics = performance_metrics(train_result['strategy_returns'])

        test_result = run_pair_trading(test_df['TSMC'], test_df['MTC'], beta=beta, **best_params)
        test_metrics = performance_metrics(test_result['strategy_returns'])

        validation_results.append({
            'Train Years': f"{train_years[0]}-{train_years[-1]}",
            'Test Years': f"{test_years[0]}-{test_years[-1]}",
            'Beta': beta,
            'Window': best_params['window'],
            'EntryZ': best_params['entry_z'],
            'ExitZ': best_params['exit_z'],
            'Train Annualized Return': train_metrics['Annualized Return'],
            'Train Volatility': train_metrics['Annualized Volatility'],
            'Train Sharpe': train_metrics['Sharpe Ratio'],
            'Train Cumulative Return': train_metrics['Cumulative Return'],
            'Train Max Drawdown': train_metrics['Max Drawdown'],
            'Test Annualized Return': test_metrics['Annualized Return'],
            'Test Volatility': test_metrics['Annualized Volatility'],
            'Test Sharpe': test_metrics['Sharpe Ratio'],
            'Test Cumulative Return': test_metrics['Cumulative Return'],
            'Test Max Drawdown': test_metrics['Max Drawdown'],
        })

    return pd.DataFrame(validation_results)


# 自訂每月 IRR 計算函數（備援用），目前實際並未在 main 中使用
# 這個函數可用於計算現金流的月度內部收益率
def monthly_irr(cashflows):
    cashflows = np.array(cashflows, dtype=float)
    coeffs = cashflows[::-1]
    roots = np.roots(coeffs)
    real_roots = np.real(roots[np.isreal(roots)])
    positive_roots = real_roots[real_roots > 0]
    if len(positive_roots) == 0:
        return np.nan
    x = positive_roots[np.argmin(np.abs(positive_roots - 1))]
    return x - 1

# 印出策略績效指標的輔助函數
# 支援百分比與數值格式化輸出，方便快速檢視
def print_metrics(name, metrics):
    print(f"=== {name} ===")
    for k, v in metrics.items():
        if np.isfinite(v):
            if 'Return' in k or 'Drawdown' in k:
                print(f"{k}: {v:.2%}")
            else:
                print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print()


def main():
    # 讀取資料並顯示日期範圍與前幾筆資料
    df = load_data()
    print('資料範圍：', df.index.min().date(), '到', df.index.max().date())
    print(df.head())

    # 初始資本
    initial_capital = 10_000

    # Pairs trading 策略回測
    pair_df = run_pair_trading(df['TSMC'], df['MTC'], notional=initial_capital)
    pair_metrics = performance_metrics(pair_df['strategy_returns'])

    # Buy-and-hold 策略：等權分配初始資金
    bh_shares_tsmc = initial_capital / 2 / df['TSMC'].iloc[0]
    bh_shares_mtc = initial_capital / 2 / df['MTC'].iloc[0]
    bh_value = bh_shares_tsmc * df['TSMC'] + bh_shares_mtc * df['MTC']
    bh_returns = bh_value.pct_change().fillna(0)
    bh_metrics = performance_metrics(bh_returns)

    # DCA 策略：每月固定投資金額
    dca_value = dca_strategy(df['TSMC'], df['MTC'], monthly_contribution=1_000)
    total_contributions = 1_000 * len(dca_value.resample('M'))
    dca_simple_return = dca_value.iloc[-1] / total_contributions - 1
    
    # 改进DCA績效計算：計算投資期間的月度收益（只在投資月計算收益）
    dca_monthly_values = dca_value.resample('M').last()
    dca_monthly_returns = dca_monthly_values.pct_change().fillna(0)
    dca_perf = performance_metrics_monthly(dca_monthly_returns)
    
    # 現金流法計算IRR
    dca_cashflows = [-1_000] * (len(dca_value.resample('M')) - 1) + [-1_000 + dca_value.iloc[-1]]
    dca_monthly_irr = npf.irr(dca_cashflows)
    dca_annual_irr = (1 + dca_monthly_irr) ** 12 - 1 if not np.isnan(dca_monthly_irr) else np.nan
    dca_metrics = {
        'Annualized Return': dca_annual_irr,
        'Annualized Volatility': dca_perf['Annualized Volatility'],
        'Sharpe Ratio': dca_perf['Sharpe Ratio'],
        'Cumulative Return': dca_simple_return,
        'Max Drawdown': dca_perf['Max Drawdown'],
        'Final Wealth': dca_value.iloc[-1],
    }

    # 印出三種策略的主要績效
    print_metrics('Pairs Trading', pair_metrics)
    print(f'Total contributions: ${initial_capital:,.0f}')
    print(f'Final portfolio value: ${pair_df["strategy_value"].iloc[-1]:,.0f}')
    print(f'Simple cumulative return after contributions: {(pair_df["strategy_value"].iloc[-1] / initial_capital - 1):.2%}')
    print(f'Approximate annual IRR: {pair_metrics["Annualized Return"]:.2%}')
    print()
    print_metrics('Buy and Hold', bh_metrics)
    print(f'Total contributions: ${initial_capital:,.0f}')
    print(f'Final portfolio value: ${bh_value.iloc[-1]:,.0f}')
    print(f'Simple cumulative return after contributions: {(bh_value.iloc[-1] / initial_capital - 1):.2%}')
    print(f'Approximate annual IRR: {bh_metrics["Annualized Return"]:.2%}')
    print()
    print_metrics('DCA', dca_metrics)
    print(f'Total contributions: ${total_contributions:,.0f}')
    print(f'Final portfolio value: ${dca_value.iloc[-1]:,.0f}')
    print(f'Simple cumulative return after contributions: {dca_simple_return:.2%}')
    print(f'Approximate annual IRR: {dca_annual_irr:.2%}')
    print()

    # 比較 Pairs Trading、Buy-and-Hold 與 DCA 的核心績效指標
    metrics_df = pd.DataFrame({
        'Pairs Trading': pair_metrics,
        'Buy and Hold': bh_metrics,
        'DCA': dca_metrics,
    }).T
    comparison_table = metrics_df[['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Cumulative Return', 'Max Drawdown']]
    print(comparison_table)
    comparison_table.to_csv('strategy_comparison.csv')
    print('\n策略比較已輸出到 strategy_comparison.csv')

    # Rolling temporal validation：使用滾雪球式訓練/測試視窗逐步擴展訓練期
    validation_df = rolling_temporal_validation(df)
    print('\nRolling Temporal Validation Results:')
    if not validation_df.empty:
        print(f'Validation windows: {len(validation_df)}')
        validation_display = validation_df[['Train Years', 'Test Years', 'Window', 'EntryZ', 'ExitZ', 'Test Annualized Return', 'Test Volatility', 'Test Sharpe', 'Test Cumulative Return', 'Test Max Drawdown']]
        print(validation_display)
        validation_df.to_csv('temporal_validation_results.csv', index=False)
        print('\n時序驗證結果已輸出到 temporal_validation_results.csv')
        avg_test = validation_df[['Test Annualized Return', 'Test Volatility', 'Test Sharpe', 'Test Cumulative Return', 'Test Max Drawdown']].mean()
        print('\nAverage test-period metrics across windows:')
        for metric, value in avg_test.items():
            if 'Return' in metric or 'Drawdown' in metric:
                print(f'{metric}: {value:.2%}')
            else:
                print(f'{metric}: {value:.4f}')
    else:
        print('No temporal validation windows available.')

    # 繪製 z-score 交易信號圖並儲存成檔案
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pair_df.index, pair_df['zscore'], label='Spread z-score')
    ax.axhline(2.5, color='red', linestyle='--', alpha=0.7, label='Entry threshold ±2.5')
    ax.axhline(-2.5, color='red', linestyle='--', alpha=0.7)
    ax.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='Exit threshold ±0.2')
    ax.axhline(-0.2, color='orange', linestyle='--', alpha=0.7)
    ax.set_title('Pairs Trading Spread z-score')
    ax.set_ylabel('Z-score')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Figure_zscore.png', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()
