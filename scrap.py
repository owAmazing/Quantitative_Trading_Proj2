import requests
import pandas as pd
from datetime import datetime
import time

# ====================== 設定 ======================
START_DATE = "2005-01-01"

# 五檔股票
tickers = ['2330.TW', '2454.TW', 'AAPL', 'PEP', 'KO']

# ====================== 下載函數 ======================
def download_stock(ticker):
    print(f"正在下載 {ticker} ...")
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    
    params = {
        "period1": int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp()),
        "period2": int(datetime.now().timestamp()),
        "interval": "1d",
        "events": "history"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()  # 檢查 HTTP 錯誤
        
        data = resp.json()
        result = data['chart']['result'][0]
        
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'date': pd.to_datetime(timestamps, unit='s').date,
            'open': quotes['open'],
            'high': quotes['high'],
            'low': quotes['low'],
            'close': quotes['close'],
            'volume': quotes['volume']
        })
        
        df.set_index('date', inplace=True)
        filename = f"{ticker.replace('.','_')}_20y.csv"
        df.to_csv(filename)
        
        print(f"✓ {ticker} 下載成功！共 {len(df)} 筆資料")
        print(f"   期間：{df.index[0]} ~ {df.index[-1]}\n")
        
        time.sleep(1)  # 避免請求太頻繁被擋
        return df
        
    except Exception as e:
        print(f"✗ {ticker} 下載失敗: {e}\n")
        return None


# ====================== 主程式 ======================
if __name__ == "__main__":
    print("=== 開始下載 5 檔股票 20 年歷史資料 ===\n")
    
    for ticker in tickers:
        download_stock(ticker)
    
    print("="*70)
    print("所有下載完成！最終檔案檢查：")
    print("="*70)
    
    for ticker in tickers:
        filename = f"{ticker.replace('.','_')}_20y.csv"
        try:
            df = pd.read_csv(filename)
            print(f"{ticker:12} : {len(df):5} 筆資料  |  {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        except:
            print(f"{ticker:12} : ❌ 檔案不存在或下載失敗")