# metrics_calc.py
import numpy as np
import pandas as pd

def calculate_all_metrics(equity_series, return_series, date_series):
    """
    自動根據日期頭尾計算精確年數，並輸出專案要求的四個核心衡量指標
    
    參數:
    equity_series: 每日資產淨值序列 (Equity Curve)
    return_series: 每日策略報酬率序列 (Strategy Returns)
    date_series: 每日日期序列 (Datetime Series)
    """
    # 1. Cumulative Return (累積報酬率)
    cum_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    
    # 2. 用最後一個資料的日期減去第一個抓取的資料日期，計算精確年數 (自動適應閏年)
    total_days_delta = date_series.max() - date_series.min()
    years = total_days_delta.days / 365.25
    
    # 安全機制：防止極端狀況（如資料不足一年導致除以 0）
    if years <= 0:
        years = 1 / 252
        
    # 計算 Annualized Return (年化報酬率 - CAGR 複利公式)
    annual_return = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1
    
    # 3. Maximum Drawdown (最大回撤)
    running_max = equity_series.cummax()
    drawdown = (running_max - equity_series) / running_max
    mdd = drawdown.max()
    
    # 4. Volatility (年化波動率)
    # 計算每日報酬率的標準差，並乘以一年交易日(252天)的開根號進行年化
    volatility = return_series.std() * (252 ** 0.5)
    
    # 回傳計算結果字典
    return {
        "Cumulative_Return": cum_return,
        "Annualized_Return": annual_return,
        "Maximum_Drawdown": mdd,
        "Volatility": volatility
    }