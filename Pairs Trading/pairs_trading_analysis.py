import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# 讀取 KO 與 PEP 的歷史收盤價資料，並對齊共同交易日
# 返回單一 DataFrame，索引為交易日期，包含 KO 與 PEP 收盤價
def load_data():
    ko = pd.read_csv('KO_20y.csv', parse_dates=['Date'])
    pep = pd.read_csv('PEP_20y.csv', parse_dates=['Date'])
    ko = ko[['Date', 'Close']].rename(columns={'Close': 'KO'})
    pep = pep[['Date', 'Close']].rename(columns={'Close': 'PEP'})
    df = pd.merge(ko, pep, on='Date', how='inner').sort_values('Date').set_index('Date')
    df = df.dropna()
    df = df.loc[df.index >= pd.Timestamp('2005-01-01')]
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

    # 依照訓練期 beta 計算價差：KO - beta * PEP
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

    # 根據當前部位計算每檔股票的股數
    # A 股數量 = total capital / (A_price + beta * B_price)
    # B 股數量 = beta * A_shares（做空時加負號）
    current_a = 0.0
    current_b = 0.0
    for idx, row in df.iterrows():
        if row['position'] == 1:
            base_shares = notional / (row['A'] + row['beta'] * row['B'])
            current_a = base_shares
            current_b = -row['beta'] * base_shares
        elif row['position'] == -1:
            base_shares = notional / (row['A'] + row['beta'] * row['B'])
            current_a = -base_shares
            current_b = row['beta'] * base_shares
        else:
            current_a = 0.0
            current_b = 0.0

        df.at[idx, 'a_shares'] = current_a
        df.at[idx, 'b_shares'] = current_b

    # 計算部位價值與策略每日報酬
    # 利用昨天的持股股數 (shift(1)) 乘以今天的價格變動 (diff())
    df['daily_pnl'] = df['a_shares'].shift(1) * df['A'].diff() + df['b_shares'].shift(1) * df['B'].diff()
    
    # 每日報酬率 = 每日損益 / 初始總資金
    df['strategy_returns'] = df['daily_pnl'].fillna(0) / notional
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

# 定期定額 (DCA) 策略：每月月底分批平均買進兩檔股票
# 這裡將每月 1,000 USD 均分成 KO 與 PEP
def dca_strategy(prices_a, prices_b, monthly_contribution=1_000):
    df = pd.DataFrame({'A': prices_a, 'B': prices_b})
    month_ends = df.resample('ME').last().index
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
def _annualized_lump_sum_irr(value_series, initial_value):
    """計算單次投資的年化IRR。value_series應為Series（帶時間索引）。"""
    if len(value_series) < 2 or initial_value <= 0:
        return np.nan
    final_value = float(value_series.iloc[-1])
    if final_value <= 0:
        return np.nan
    days = (value_series.index[-1] - value_series.index[0]).days
    years = days / 365.25
    if years <= 0:
        return np.nan
    return (final_value / initial_value) ** (1 / years) - 1


def _compute_dca_annual_irr(prices_a, prices_b, monthly_contribution=1_000):
    value = dca_strategy(prices_a, prices_b, monthly_contribution=monthly_contribution)
    monthly_values = value.resample('ME').last()
    cashflows = [-monthly_contribution] * len(monthly_values)
    if not cashflows:
        return np.nan
    # 最後一個月的現金流 = -投資 + 月底投資組合價值
    cashflows[-1] += monthly_values.iloc[-1]
    irr_monthly = npf.irr(cashflows)
    if np.isnan(irr_monthly):
        return np.nan
    return (1 + irr_monthly) ** 12 - 1


def rolling_temporal_validation(df):
    years = sorted(df.index.year.unique())
    # Exclude incomplete year (2026 only has data up to May 15)
    years = [y for y in years if y < 2026]
    
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

        beta = estimate_hedge_ratio(train_df['KO'], train_df['PEP'])
        best_params = find_best_params(train_df['KO'], train_df['PEP'], beta)

        train_result = run_pair_trading(train_df['KO'], train_df['PEP'], beta=beta, **best_params)
        train_metrics = performance_metrics(train_result['strategy_returns'])

        test_result = run_pair_trading(test_df['KO'], test_df['PEP'], beta=beta, **best_params)
        test_metrics = performance_metrics(test_result['strategy_returns'])

        test_bh_shares_ko = 10_000 / 2 / test_df['KO'].iloc[0]
        test_bh_shares_pep = 10_000 / 2 / test_df['PEP'].iloc[0]
        test_bh_value = test_bh_shares_ko * test_df['KO'] + test_bh_shares_pep * test_df['PEP']
        test_bh_irr = _annualized_lump_sum_irr(test_bh_value, 10_000)

        test_dca_irr = _compute_dca_annual_irr(test_df['KO'], test_df['PEP'], monthly_contribution=1_000)

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
            'Test IRR Pairs Trading': _annualized_lump_sum_irr(test_result['strategy_value'], 10_000),
            'Test IRR Buy and Hold': test_bh_irr,
            'Test IRR DCA': test_dca_irr,
        })

    return pd.DataFrame(validation_results)


def plot_temporal_validation(validation_df, out_filepath='temporal_validation.png'):
    """Plot temporal validation results in 2x2 panels similar to reference.

    Panels:
    - Test Cumulative Return (%)
    - Test Annualized Return (%)
    - Test Max Drawdown (%)
    - Test Volatility (%)
    """
    if validation_df.empty:
        print('No validation results to plot.')
        return

    # x labels: use Test Years for each validation window
    labels = validation_df['Test Years'].astype(str).tolist()
    x = list(range(len(labels)))

    cum = validation_df['Test Cumulative Return'] * 100
    ann = validation_df['Test Annualized Return'] * 100
    mdd = validation_df['Test Max Drawdown'] * 100
    vol = validation_df['Test Volatility'] * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temporal Validation Results', fontsize=16)

    ax = axes[0, 0]
    ax.plot(x, cum, marker='o', linestyle='-', color='tab:blue')
    ax.set_title('Test Cumulative Return (%)')
    ax.set_ylabel('Cumulative Return (%)')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[0, 1]
    ax.plot(x, ann, marker='s', linestyle='-', color='tab:orange')
    ax.set_title('Test Annualized Return (%)')
    ax.set_ylabel('Annualized Return (%)')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1, 0]
    ax.plot(x, mdd, marker='d', linestyle='-', color='tab:green')
    ax.set_title('Test Maximum Drawdown (%)')
    ax.set_ylabel('Max Drawdown (%)')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1, 1]
    ax.plot(x, vol, marker='^', linestyle='-', color='tab:red')
    ax.set_title('Test Volatility (%)')
    ax.set_ylabel('Volatility (%)')
    ax.grid(True, linestyle='--', alpha=0.5)

    # X ticks
    for a in axes.flatten():
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f'Temporal validation plot saved to {out_filepath}')


def plot_irr_comparison(validation_df, out_filepath='temporal_validation_irr.png'):
    """Plot IRR comparison across test periods for Pairs Trading, B&H, and DCA using line chart."""
    if validation_df.empty:
        print('No validation results to plot IRR.')
        return

    labels = validation_df['Test Years'].astype(str).tolist()
    x = list(range(len(labels)))
    pt_irr = validation_df['Test IRR Pairs Trading'] * 100
    bh_irr = validation_df['Test IRR Buy and Hold'] * 100
    dca_irr = validation_df['Test IRR DCA'] * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(x, pt_irr, marker='o', linestyle='-', linewidth=2, markersize=6, 
            color='tab:blue', label='Pairs Trading')
    ax.plot(x, bh_irr, marker='s', linestyle='-', linewidth=2, markersize=6, 
            color='tab:orange', label='Buy and Hold')
    ax.plot(x, dca_irr, marker='^', linestyle='-', linewidth=2, markersize=6, 
            color='tab:green', label='DCA')

    ax.set_title('Test-Period Annualized IRR Comparison (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Test Period', fontsize=11)
    ax.set_ylabel('Annualized IRR (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f'IRR comparison plot saved to {out_filepath}')


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
    pair_df = run_pair_trading(df['KO'], df['PEP'], notional=initial_capital)
    pair_metrics = performance_metrics(pair_df['strategy_returns'])

    # Buy-and-hold 策略：等權分配初始資金
    bh_shares_ko = initial_capital / 2 / df['KO'].iloc[0]
    bh_shares_pep = initial_capital / 2 / df['PEP'].iloc[0]
    bh_value = bh_shares_ko * df['KO'] + bh_shares_pep * df['PEP']
    bh_returns = bh_value.pct_change().fillna(0)
    bh_metrics = performance_metrics(bh_returns)

    # DCA 策略：每月固定投資金額
    dca_value = dca_strategy(df['KO'], df['PEP'], monthly_contribution=1_000)
    dca_perf = performance_metrics(dca_value.pct_change().fillna(0))
    total_contributions = 1_000 * len(dca_value.resample('ME'))
    dca_simple_return = dca_value.iloc[-1] / total_contributions - 1
    dca_cashflows = [-1_000] * (len(dca_value.resample('ME')) - 1) + [-1_000 + dca_value.iloc[-1]]
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
        validation_display = validation_df[['Train Years', 'Test Years', 'Window', 'EntryZ', 'ExitZ', 'Test Annualized Return', 'Test Volatility', 'Test Sharpe', 'Test Cumulative Return', 'Test Max Drawdown', 'Test IRR Pairs Trading', 'Test IRR Buy and Hold', 'Test IRR DCA']]
        print(validation_display)
        validation_df.to_csv('temporal_validation_results.csv', index=False)
        print('\n時序驗證結果已輸出到 temporal_validation_results.csv')
        avg_test = validation_df[['Test Annualized Return', 'Test Volatility', 'Test Sharpe', 'Test Cumulative Return', 'Test Max Drawdown', 'Test IRR Pairs Trading', 'Test IRR Buy and Hold', 'Test IRR DCA']].mean()
        print('\nAverage test-period metrics across windows:')
        for metric, value in avg_test.items():
            if 'Return' in metric or 'Drawdown' in metric or 'IRR' in metric:
                print(f'{metric}: {value:.2%}')
            else:
                print(f'{metric}: {value:.4f}')
        # Plot temporal validation summary
        try:
            plot_temporal_validation(validation_df, out_filepath='temporal_validation.png')
        except Exception as e:
            print('Failed to plot temporal validation:', e)
        try:
            plot_irr_comparison(validation_df, out_filepath='temporal_validation_irr.png')
        except Exception as e:
            print('Failed to plot IRR comparison:', e)
    else:
        print('No temporal validation windows available.')

    # 繪製 z-score 交易信號圖並儲存成檔案
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pair_df.index, pair_df['zscore'], label='Spread z-score')
    ax.set_title('Pairs Trading Spread z-score')
    ax.set_ylabel('Z-score')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Figure_zscore.png', dpi=100)


if __name__ == '__main__':
    main()
