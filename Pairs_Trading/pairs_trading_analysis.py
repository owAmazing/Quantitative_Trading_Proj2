import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


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
    df = pd.DataFrame({'A': prices_a, 'B': prices_b})

    if beta is None:
        beta = estimate_hedge_ratio(df['A'], df['B'])
    df['beta'] = beta

    df['spread'] = compute_spread(df['A'], df['B'], beta)
    df['spread_mean'] = df['spread'].rolling(window).mean()
    df['spread_std'] = df['spread'].rolling(window).std()

    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    df['zscore'] = df['zscore'].fillna(0)

    # 1. 訊號生成：當天收盤決定訊號，隔天(t+1)才執行
    target_position = 0
    target_positions = []
    for z in df['zscore']:
        if target_position == 0:
            if z > entry_z:
                target_position = -1
            elif z < -entry_z:
                target_position = 1
        elif abs(z) < exit_z:
            target_position = 0
        target_positions.append(target_position)
    
    # 將訊號 shift(1)，代表今天交易開盤/執行的部位是基於昨天的訊號
    df['position'] = pd.Series(target_positions, index=df.index).shift(1).fillna(0).astype(int)

    df['a_shares'] = 0.0
    df['b_shares'] = 0.0

    current_a = 0.0
    current_b = 0.0
    in_position = False

    # 2. 股數計算：只有在進場那一刻決定股數，持倉期間保持不變
    # 這裡使用昨天的收盤價作為 t 日執行的基準價格
    price_a_prev = df['A'].shift(1)
    price_b_prev = df['B'].shift(1)

    for idx, row in df.iterrows():
        pos = row['position']
        p_a = price_a_prev.at[idx]
        p_b = price_b_prev.at[idx]
        beta_val = row['beta']

        if pos != 0:
            if not in_position:
                # 剛進場，計算固定股數
                if pd.isna(p_a) or pd.isna(p_b):
                    current_a, current_b = 0.0, 0.0
                else:
                    base_shares = notional / (p_a + beta_val * p_b)
                    if pos == 1:
                        current_a = base_shares
                        current_b = -beta_val * base_shares
                    else:
                        current_a = -base_shares
                        current_b = beta_val * base_shares
                in_position = True
            # if in_position == True 且 pos 未變，則 current_a, current_b 沿用，不重新計算
        else:
            current_a = 0.0
            current_b = 0.0
            in_position = False

        df.at[idx, 'a_shares'] = current_a
        df.at[idx, 'b_shares'] = current_b

    # 3. 損益計算：t 日的損益 = t-1 日留下來的持股 * (t 日收盤價 - t-1 日收盤價)
    df['daily_pnl'] = df['a_shares'] * df['A'].diff() + df['b_shares'] * df['B'].diff()
    df['daily_pnl'] = df['daily_pnl'].fillna(0)
    
    # 修正回測資產淨值曲線 (不採用單純複利，而是淨值累加，避免股數未隨淨值調整的矛盾)
    df['strategy_value'] = notional + df['daily_pnl'].cumsum()
    df['strategy_returns'] = df['daily_pnl'] / notional # 單利型報酬率，用於配合固定本金回測
    
    return df.dropna()


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
    # 改用單利累積法對應固定本金策略
    cum_ret = returns.sum()
    ann_ret = cum_ret * (252 / len(returns))
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    
    wealth = 1 + returns.cumsum()
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    return {
        'Annualized Return': ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Cumulative Return': cum_ret,
        'Max Drawdown': max_dd,
        'Final Wealth': wealth.iloc[-1] * 10000, # 僅供參考
    }


def dca_strategy(prices_a, prices_b, monthly_contribution=1_000):
    df = pd.DataFrame({'A': prices_a, 'B': prices_b})
    month_ends = df.resample('ME').last().index
    a_shares = 0.0
    b_shares = 0.0
    value = []
    
    # 修正 DCA 每日真報酬率計算（排除入金干擾）
    daily_returns = [0.0]
    prev_val = 0.0
    
    for date, row in df.iterrows():
        current_val = a_shares * row['A'] + b_shares * row['B']
        
        if len(value) > 0:
            # 當天開盤前的價值就是昨天的價值，計算純因價格變動引起的報酬率
            if prev_val > 0:
                ret = (current_val - prev_val) / prev_val
            else:
                ret = 0.0
            daily_returns.append(ret)
            
        if date in month_ends:
            a_shares += monthly_contribution / 2 / row['A']
            b_shares += monthly_contribution / 2 / row['B']
            # 更新投入後的真實市值
            current_val = a_shares * row['A'] + b_shares * row['B']
            
        value.append(current_val)
        prev_val = current_val
        
    return pd.Series(value, index=df.index), pd.Series(daily_returns, index=df.index)


def _annualized_lump_sum_irr(value_series, initial_value):
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
    value, _ = dca_strategy(prices_a, prices_b, monthly_contribution=monthly_contribution)
    monthly_values = value.resample('ME').last()
    cashflows = [-monthly_contribution] * len(monthly_values)
    if not cashflows:
        return np.nan
    cashflows[-1] += monthly_values.iloc[-1]
    irr_monthly = npf.irr(cashflows)
    if np.isnan(irr_monthly):
        return np.nan
    return (1 + irr_monthly) ** 12 - 1


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


def rolling_temporal_validation(df):
    years = sorted(df.index.year.unique())
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
    if validation_df.empty:
        print('No validation results to plot.')
        return

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

    for a in axes.flatten():
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_filepath, dpi=150)
    plt.close(fig)
    print(f'Temporal validation plot saved to {out_filepath}')


def plot_irr_comparison(validation_df, out_filepath='temporal_validation_irr.png'):
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
    df = load_data()
    print('資料範圍：', df.index.min().date(), '到', df.index.max().date())

    initial_capital = 10_000

    pair_df = run_pair_trading(df['KO'], df['PEP'], notional=initial_capital)
    pair_metrics = performance_metrics(pair_df['strategy_returns'])

    bh_shares_ko = initial_capital / 2 / df['KO'].iloc[0]
    bh_shares_pep = initial_capital / 2 / df['PEP'].iloc[0]
    bh_value = bh_shares_ko * df['KO'] + bh_shares_pep * df['PEP']
    bh_returns = bh_value.pct_change().fillna(0)
    bh_metrics = performance_metrics(bh_returns)

    dca_value, dca_daily_rets = dca_strategy(df['KO'], df['PEP'], monthly_contribution=1_000)
    dca_perf = performance_metrics(dca_daily_rets)
    total_contributions = 1_000 * len(dca_value.resample('ME'))
    dca_simple_return = dca_value.iloc[-1] / total_contributions - 1
    
    dca_cashflows = [-1_000] * (len(dca_value.resample('ME')) - 1) + [-1_000 + dca_value.iloc[-1]]
    dca_monthly_irr = npf.irr(dca_cashflows)
    dca_annual_irr = (1 + dca_monthly_irr) ** 12 - 1 if not np.isnan(dca_monthly_irr) else np.nan
    
    dca_metrics = {
        'Annualized Return': dca_annual_irr, # 對於定期定額，IRR 較適合作為年化報酬率標準
        'Annualized Volatility': dca_perf['Annualized Volatility'],
        'Sharpe Ratio': dca_annual_irr / dca_perf['Annualized Volatility'] if dca_perf['Annualized Volatility'] != 0 else np.nan,
        'Cumulative Return': dca_simple_return,
        'Max Drawdown': dca_perf['Max Drawdown'],
        'Final Wealth': dca_value.iloc[-1],
    }

    print_metrics('Pairs Trading', pair_metrics)
    print(f'Final portfolio value: ${pair_df["strategy_value"].iloc[-1]:,.0f}\n')
    
    print_metrics('Buy and Hold', bh_metrics)
    print(f'Final portfolio value: ${bh_value.iloc[-1]:,.0f}\n')
    
    print_metrics('DCA', dca_metrics)
    print(f'Total contributions: ${total_contributions:,.0f}')
    print(f'Final portfolio value: ${dca_value.iloc[-1]:,.0f}\n')

    metrics_df = pd.DataFrame({
        'Pairs Trading': pair_metrics,
        'Buy and Hold': bh_metrics,
        'DCA': dca_metrics,
    }).T
    comparison_table = metrics_df[['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Cumulative Return', 'Max Drawdown']]
    print(comparison_table)
    comparison_table.to_csv('strategy_comparison.csv')

    validation_df = rolling_temporal_validation(df)
    if not validation_df.empty:
        validation_df.to_csv('temporal_validation_results.csv', index=False)
        avg_test = validation_df[['Test Annualized Return', 'Test Volatility', 'Test Sharpe', 'Test Cumulative Return', 'Test Max Drawdown', 'Test IRR Pairs Trading', 'Test IRR Buy and Hold', 'Test IRR DCA']].mean()
        print('\nAverage test-period metrics across windows:')
        for metric, value in avg_test.items():
            if 'Return' in metric or 'Drawdown' in metric or 'IRR' in metric:
                print(f'{metric}: {value:.2%}')
            else:
                print(f'{metric}: {value:.4f}')
        try:
            plot_temporal_validation(validation_df, out_filepath='temporal_validation.png')
        except Exception as e:
            print('Failed to plot temporal validation:', e)
        try:
            plot_irr_comparison(validation_df, out_filepath='temporal_validation_irr.png')
        except Exception as e:
            print('Failed to plot IRR comparison:', e)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pair_df.index, pair_df['zscore'], label='Spread z-score')
    ax.set_title('Pairs Trading Spread z-score')
    ax.set_ylabel('Z-score')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Figure_zscore.png', dpi=100)


if __name__ == '__main__':
    main()