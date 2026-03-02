"""
创业板ETF监控与回测系统
功能：
1. 策略收益 vs 买入持有收益对比
2. 实时行情获取
3. 买卖信号提示
4. 历史回测分析

依赖：streamlit, pandas, numpy, requests, plotly, akshare, baostock, pypinyin
安装：pip install streamlit pandas numpy requests plotly akshare baostock pypinyin
运行：streamlit run etf_monitor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import os
import glob
import json as pyjson

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    import stock_prediction_module as spm
except ImportError as e:
    spm = None

try:
    import stock_analysis_module as sam
except ImportError as e:
    sam = None

try:
    import stock_selection_module as ssm
except ImportError as e:
    ssm = None

# ============================================
# 配置
# ============================================

# 中国法定节假日和周末（2024-2026年）
# 用于K线图和交易日过滤
CHINA_HOLIDAYS = {
    # 元旦
    "2024-01-01",
    "2025-01-01",
    "2026-01-01",
    # 春节
    "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31",
    # 清明
    "2024-04-04", "2024-04-05", "2024-04-06",
    "2025-04-04", "2025-04-05", "2025-04-06",
    "2026-04-04", "2026-04-05", "2026-04-06",
    # 劳动节
    "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    # 端午
    "2024-06-10", "2025-06-02",
    "2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13",
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05",
    "2026-10-06", "2026-10-07", "2026-10-06", "2026-10-06", "2026-10-06", "2026-10-06",
    # 中秋
    "2024-09-15", "2024-09-16", "2024-09-17",
    "2025-09-14", "2025-09-15", "2025-09-16",
    "2026-09-15", "2026-09-16", "2026-09-17",
    # 国庆
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07",
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07",
}

def is_trading_day(date):
    """判断是否为交易日（非周末且非节假日）"""
    # 转换为字符串格式
    if isinstance(date, pd.Timestamp):
        date_str = date.strftime('%Y-%m-%d')
    else:
        date_str = str(date)

    # 检查是否为周末（周一到周五为交易日）
    weekday = pd.to_datetime(date_str).weekday()
    if weekday >= 5:  # 周六(5) 或 周日(6)
        return False

    # 检查是否为法定节假日
    if date_str in CHINA_HOLIDAYS:
        return False

    return True

st.set_page_config(
    page_title="创业板ETF监控",
    page_icon="📈",
    layout="wide"
)

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.stop()

# ============================================
# 缓存工具函数：按日期缓存，避免同一天重复操作
# ============================================

def get_today_str():
    """获取今天日期字符串（YYYYMMDD格式）"""
    return datetime.now().strftime("%Y%m%d")

def is_cache_valid(cache_key, today_str=None):
    """检查缓存是否有效（当天）"""
    if today_str is None:
        today_str = get_today_str()
    cache_date_key = f"{cache_key}_date"
    cached_date = st.session_state.get(cache_date_key, "")
    return cached_date == today_str

def set_cache(cache_key, data, today_str=None):
    """设置缓存，并记录日期"""
    if today_str is None:
        today_str = get_today_str()
    cache_date_key = f"{cache_key}_date"
    st.session_state[cache_key] = data
    st.session_state[cache_date_key] = today_str

def get_cache(cache_key):
    """获取缓存数据"""
    return st.session_state.get(cache_key)

def clear_day_cache(cache_key):
    """清除指定缓存"""
    cache_date_key = f"{cache_key}_date"
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    if cache_date_key in st.session_state:
        del st.session_state[cache_date_key]

# ============================================
# 扩展功能：多股票支持与策略优化
# ============================================

def optimize_strategy_for_stock(df, timeframe="日线"):
    """
    针对特定股票数据自动优化策略参数
    基于 optimize_daily_v2 的逻辑简化版
    """
    if df is None or len(df) < 200:
        return None
    
    # 缩小搜索空间以提高响应速度
    if timeframe == "日线":
        # 用户自定义扩展参数空间
        atr_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
        lo_shorts = [15, 20, 25, 30, 35, 40, 45]
        lo_longs = [50, 60, 75, 90, 100]
        hi_shorts = [5, 10, 15, 20]
        hi_longs = [10, 15, 20, 25]
        rsi_lows = [30, 35, 40, 45, 50]
        rsi_highs = [70, 75, 80, 85, 90]
    else: # 30分钟
        atr_thresholds = [1.0, 1.3, 1.5]
        lo_shorts = [15, 25, 35]
        lo_longs = [60, 80]
        hi_shorts = [5, 10]
        hi_longs = [15, 20]
        rsi_lows = [40, 45]
        rsi_highs = [70, 80]

    # 预计算指标
    all_mas = set(lo_shorts + lo_longs + hi_shorts + hi_longs)
    df_calc = calculate_indicators(df.copy(), ma_lengths=list(all_mas))
    
    best_ret = -999
    best_params = None
    
    # 使用随机采样或有限网格搜索
    # 这里使用全网格但参数较少
    from itertools import product
    
    # 简单的回测函数 (内部类，向量化优化)
    def fast_backtest(d, atr, hip, lop, rl, rh):
        # 1. 识别波动率状态
        is_high = d['ATR_Pct'].values > atr
        
        # 2. 获取均线数据 (预先提取为numpy array)
        hi_s = d[f'MA{hip[0]}'].values
        hi_l = d[f'MA{hip[1]}'].values
        lo_s = d[f'MA{lop[0]}'].values
        lo_l = d[f'MA{lop[1]}'].values
        
        # 3. 构建动态均线
        s_ma = np.where(is_high, hi_s, lo_s)
        l_ma = np.where(is_high, hi_l, lo_l)
        
        close = d['Close'].values
        rsi = d['RSI'].values
        roc = d['ROC20'].values
        
        # 4. 生成信号矩阵 (Buy=1, Sell=-1, Hold=0)
        # Buy: Close > Short > Long & RSI in [L, H] & ROC > 0
        buy_sig = (close > s_ma) & (s_ma > l_ma) & (rsi >= rl) & (rsi <= rh) & (roc > 0)
        
        # Sell: Close < Short | Short < Long | RSI > H+10 | RSI < L-10
        # 优化点：对于强势股，RSI超买可能不应直接卖出，这里增加一个宽松模式的测试
        # 但在通用回测中，我们先保持标准逻辑，或者后续引入 rsi_sell_enabled 参数
        r_up = min(100, rh + 10)
        r_low = max(0, rl - 10)
        sell_sig = (close < s_ma) | (s_ma < l_ma) | (rsi > r_up) | (rsi < r_low)
        
        # 5. 向量化持仓计算
        # 信号：1=买入, -1=卖出, 0=无信号
        signals = np.zeros(len(d))
        signals[buy_sig] = 1
        signals[sell_sig] = -1
        
        # 状态机：ffill传播持仓状态
        # 技巧：将0替换为NaN，然后ffill，再填回0。
        # 1 (Buy) -> 1 (Hold) ... -> -1 (Sell) -> -1 (Empty) ...
        # 我们需要的是：遇到1变持仓(1)，遇到-1变空仓(0)。
        # 我们可以用一个累积最大值或者pandas的ffill
        # 但要注意：今天出的信号，明天才能操作（或者收盘操作）。
        # 这里假设收盘价成交（Backtest逻辑）。
        
        # 简单的Python循环虽然慢，但对于复杂的状态转换（如持仓时才卖出）最准确。
        # 纯向量化处理 "持仓保持" 比较麻烦，需要 pandas ffill。
        # 混合法：
        
        # 转换为 Series 以使用 ffill
        sig_series = pd.Series(signals)
        # 将 0 (无信号) 替换为 NaN，保留 1 (买) 和 -1 (卖)
        sig_series = sig_series.replace(0, np.nan)
        # 前向填充：1 ... 1 ... -1 ... -1
        # 填充后的 1 代表持仓，-1 代表空仓
        # 初始填充设为 -1 (空仓)
        sig_series.iloc[0] = -1 if pd.isna(sig_series.iloc[0]) else sig_series.iloc[0]
        pos_series = sig_series.ffill()
        
        # position: 1=持仓, 0=空仓
        # 如果当前是 1，则持有；如果是 -1，则空仓
        position = (pos_series == 1).astype(int).values
        
        # 计算收益
        # 策略收益 = 持仓状态 * 标的涨跌幅
        # 注意：如果在 i 时刻发出买入信号（收盘），则持有区间是从 i 到 i+1 吗？
        # 原始逻辑：if buy[i] -> in_pos=True, entry=close[i]. 收益从 i+1 开始计算。
        # 所以 position 数组需要 shift(1)
        
        pct_change = d['Close'].pct_change().fillna(0).values
        # pos[i] 表示第 i 天收盘时的持仓状态。
        # 第 i+1 天的收益取决于 pos[i]
        strat_ret = position[:-1] * pct_change[1:]
        
        # 补齐长度
        strat_ret = np.insert(strat_ret, 0, 0)
        
        # 考虑预热期 (前60天不操作)
        strat_ret[:60] = 0
        
        # 累积收益
        cum_ret = np.prod(1 + strat_ret) - 1
        return cum_ret * 100

    # 运行搜索 (使用进度条)
    import time
    total_combs = len(atr_thresholds) * len(list(product(lo_shorts, lo_longs))) * len(list(product(hi_shorts, hi_longs))) * len(list(product(rsi_lows, rsi_highs)))
    # st.write(f"正在遍历 {total_combs} 种参数组合...")
    
    # 增加参数：是否放宽RSI卖出限制（针对大牛股）
    # 简单实现：尝试两轮，一轮标准，一轮放宽RSI上限
    
    params_list = []
    
    # 构建参数列表
    for atr in atr_thresholds:
        for ls, ll in product(lo_shorts, lo_longs):
            if ls >= ll: continue
            for hs, hl in product(hi_shorts, hi_longs):
                if hs >= hl: continue
                for rl, rh in product(rsi_lows, rsi_highs):
                    if rl >= rh: continue
                    params_list.append((atr, (hs, hl), (ls, ll), rl, rh))
    
    # 批量测试
    for p in params_list:
        atr, hip, lop, rl, rh = p
        ret = fast_backtest(df_calc, atr, hip, lop, rl, rh)
        if ret > best_ret:
            best_ret = ret
            best_params = {
                "atr_threshold": atr,
                "hi_short": hip[0], "hi_long": hip[1],
                "lo_short": lop[0], "lo_long": lop[1],
                "rsi_low": rl, "rsi_high": rh,
                "return": ret
            }
            
    # 额外检查：如果 Buy&Hold 收益更高，尝试寻找更激进的参数
    # 计算 B&H 收益
    bh_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    if best_ret < bh_ret:
        # 尝试“趋势增强模式”：大幅放宽 RSI 卖出条件 (RSI > 95 才卖)
        # 这能捕捉超强趋势
        for p in params_list:
            atr, hip, lop, rl, rh = p
            # 强制 RSI High = 95
            ret = fast_backtest(df_calc, atr, hip, lop, rl, 95)
            if ret > best_ret:
                best_ret = ret
                best_params = {
                    "atr_threshold": atr,
                    "hi_short": hip[0], "hi_long": hip[1],
                    "lo_short": lop[0], "lo_long": lop[1],
                    "rsi_low": rl, "rsi_high": 95, # 标记为极高
                    "return": ret
                }
    
    return best_params

# 策略存储管理
STRATEGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_strategies.json")

def load_stock_strategy(code):
    if not os.path.exists(STRATEGY_FILE):
        return None
    try:
        with open(STRATEGY_FILE, "r", encoding="utf-8") as f:
            data = pyjson.load(f)
            return data.get(code)
    except:
        return None

def save_stock_strategy(code, strategy_data):
    all_data = {}
    if os.path.exists(STRATEGY_FILE):
        try:
            with open(STRATEGY_FILE, "r", encoding="utf-8") as f:
                all_data = pyjson.load(f)
        except:
            pass
    
    # 策略数据包含时间戳
    strategy_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_data[code] = strategy_data
    
    with open(STRATEGY_FILE, "w", encoding="utf-8") as f:
        pyjson.dump(all_data, f, ensure_ascii=False, indent=2)

# 创业板ETF代码 (默认)
DEFAULT_CODE = "159915"

# ============================================
# 数据获取模块
# ============================================

@st.cache_data(ttl=60)  # 缓存60秒
def get_realtime_price(code):
    try:
        def get_exchange_prefix(c):
            s = str(c)
            return "sz" if s.startswith(("0", "1", "2", "3")) else "sh"
        prefix = get_exchange_prefix(code)
        url = f"http://hq.sinajs.cn/list={prefix}{code}"
        headers = {"Referer": "http://finance.sina.com.cn"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.text
            quote = data.split('"')[1].split(',')
            if len(quote) >= 10:
                name = quote[0]
                open_p = float(quote[1]) if quote[1] else np.nan
                prev_close = float(quote[2]) if quote[2] else np.nan
                current = float(quote[3]) if quote[3] else np.nan
                high = float(quote[4]) if quote[4] else np.nan
                low = float(quote[5]) if quote[5] else np.nan
                volume = int(float(quote[8])) if quote[8] else 0
                amount = float(quote[9]) if quote[9] else 0.0
                ts = ""
                if len(quote) > 31 and quote[30] and quote[31]:
                    ts = quote[30] + " " + quote[31]
                return {
                    "code": code,
                    "name": name,
                    "open": open_p,
                    "pre_close": prev_close,
                    "high": high,
                    "low": low,
                    "close": current,
                    "volume": volume,
                    "amount": amount,
                    "time": ts
                }
    except Exception as e:
        st.error(f"获取实时数据失败: {e}")
    return None


def get_history_data_daily(code, days=3000, force_refresh=False):
    """获取日线数据，带当天缓存机制"""
    cache_key = f"daily_data_{code}_{days}"

    # 检查缓存（当天有效）
    if not force_refresh and is_cache_valid(cache_key):
        cached_data = get_cache(cache_key)
        if cached_data is not None:
            st.info(f"📦 使用今日缓存的日线数据 ({code})")
            return cached_data

    # 获取新数据
    df = _fetch_history_data_daily_impl(code, days)

    # 存入缓存
    if df is not None and not df.empty:
        set_cache(cache_key, df)

    return df

@st.cache_data(ttl=3600)
def _fetch_history_data_daily_impl(code, days=3000):
    """实际获取日线数据的实现"""
    try:
        import akshare as ak
        # 优先读取本地全量文件
        base_dir = os.path.dirname(os.path.abspath(__file__))
        explicit_path = os.path.join(base_dir, f"Daily_{code}_*.xlsx")
        paths = glob.glob(explicit_path)
        
        df_local = pd.DataFrame()
        if paths:
            # 找最新的一个
            paths.sort(key=os.path.getmtime, reverse=True)
            try:
                df_local = pd.read_excel(paths[0], engine="openpyxl")
                if "Date" in df_local.columns:
                    df_local["Date"] = pd.to_datetime(df_local["Date"])
            except:
                pass
        
        # 获取在线更新
        def get_exchange_prefix(c):
            s = str(c)
            return "sz" if s.startswith(("0", "1", "2", "3")) else "sh"
        prefix = get_exchange_prefix(code)
        symbol = f"{prefix}{code}"
        
        df_api = None
        try:
            # 尝试获取增量
            start_date = "20100101"
            if not df_local.empty:
                start_date = df_local["Date"].max().strftime("%Y%m%d")
            
            # 使用 ak.stock_zh_a_hist 比较稳定
            df_api = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            )
        except Exception:
            df_api = None
            
        # 合并
        if df_api is not None and not df_api.empty:
            col_map = {
                "日期": "Date", "date": "Date",
                "开盘": "Open", "open": "Open",
                "收盘": "Close", "close": "Close",
                "最高": "High", "high": "High",
                "最低": "Low", "low": "Low",
                "成交量": "Volume", "volume": "Volume",
                "成交额": "Amount", "amount": "Amount",
            }
            df_api = df_api.rename(columns={c: col_map[c] for c in df_api.columns if c in col_map})
            df_api["Date"] = pd.to_datetime(df_api["Date"])
            for c in ["Open", "Close", "High", "Low", "Volume", "Amount"]:
                if c in df_api.columns:
                    df_api[c] = pd.to_numeric(df_api[c], errors="coerce")
            
            if not df_local.empty:
                df = pd.concat([df_local, df_api], ignore_index=True)
                df = df.drop_duplicates(subset=["Date"], keep="last")
            else:
                df = df_api
        else:
            df = df_local
            
        if df is None or df.empty:
             # Fallback if everything fails
             # fund_etf_hist_sina 需要带 sh/sz 前缀的代码
             df = ak.fund_etf_hist_sina(symbol=f"{prefix}{code}")
             
        # Final cleanup
        if df is not None and not df.empty:
            # 标准化列名（小写转大写）
            col_rename = {}
            for col in df.columns:
                if col.lower() == 'date' and col != 'Date':
                    col_rename[col] = 'Date'
                elif col.lower() == 'open' and col != 'Open':
                    col_rename[col] = 'Open'
                elif col.lower() == 'close' and col != 'Close':
                    col_rename[col] = 'Close'
                elif col.lower() == 'high' and col != 'High':
                    col_rename[col] = 'High'
                elif col.lower() == 'low' and col != 'Low':
                    col_rename[col] = 'Low'
                elif col.lower() == 'volume' and col != 'Volume':
                    col_rename[col] = 'Volume'
                elif col.lower() == 'amount' and col != 'Amount':
                    col_rename[col] = 'Amount'
            if col_rename:
                df = df.rename(columns=col_rename)

            if "Date" not in df.columns and "date" in df.columns:
                 df = df.rename(columns={"date": "Date"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            start_dt = datetime.now() - timedelta(days=days)
            df = df[df["Date"] >= start_dt].reset_index(drop=True)
            # 确保所有需要的列都存在，如果不存在则创建空列
            required_cols = ["Date", "Open", "Close", "High", "Low", "Volume", "Amount"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            return df[required_cols]

        return None
    except Exception as e:
        st.error(f"获取历史数据失败: {e}")
        return None


def get_history_data_30min(code, force_refresh=False):
    """获取30分钟数据，带当天缓存机制"""
    cache_key = f"30min_data_{code}"

    # 检查缓存（当天有效）
    if not force_refresh and is_cache_valid(cache_key):
        cached_data = get_cache(cache_key)
        if cached_data is not None:
            st.info(f"📦 使用今日缓存的30分钟数据 ({code})")
            return cached_data

    # 获取新数据
    df = _fetch_history_data_30min_impl(code)

    # 存入缓存
    if df is not None and not df.empty:
        set_cache(cache_key, df)

    return df

@st.cache_data(ttl=3600)
def _fetch_history_data_30min_impl(code):
    # 1. 加载本地Excel/CSV数据 (历史基准)
    # 支持加载多个历史文件并合并
    df_local = pd.DataFrame()
    
    # NEW: 优先检查 "数据/股票数据" 或 "数据/基金数据" 目录下的 CSV
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "数据")
        sub_dirs = ["股票数据", "基金数据"]
        
        target_csvs = []
        for sub in sub_dirs:
            # 匹配包含代码的csv文件
            pattern = os.path.join(data_dir, sub, f"*{code}*.csv")
            found = glob.glob(pattern)
            if found:
                target_csvs.extend(found)
        
        if target_csvs:
            # 取最新的文件 (虽然通常只有一个)
            target_csvs.sort(key=os.path.getmtime, reverse=True)
            fpath = target_csvs[0]
            
            try:
                # 读取CSV (GBK或UTF-8)
                try:
                    df_temp = pd.read_csv(fpath, encoding="utf-8")
                except:
                    df_temp = pd.read_csv(fpath, encoding="gbk")
                
                # 映射列名
                rename_map = {
                    "时间": "Date", "代码": "Code", "名称": "Name",
                    "开盘价": "Open", "收盘价": "Close", 
                    "最高价": "High", "最低价": "Low", 
                    "成交量": "Volume", "成交额": "Amount",
                    "涨幅": "PctChange", "振幅": "Amplitude"
                }
                # 兼容英文列名
                for col in df_temp.columns:
                    c_low = str(col).lower()
                    if c_low in ["date", "time", "day"]: rename_map[col] = "Date"
                    elif "open" in c_low: rename_map[col] = "Open"
                    elif "close" in c_low: rename_map[col] = "Close"
                    elif "high" in c_low: rename_map[col] = "High"
                    elif "low" in c_low: rename_map[col] = "Low"
                    elif "volume" in c_low: rename_map[col] = "Volume"
                    elif "amount" in c_low: rename_map[col] = "Amount"
                
                df_temp = df_temp.rename(columns=rename_map)
                
                if "Date" in df_temp.columns:
                    df_temp["Date"] = pd.to_datetime(df_temp["Date"])
                    # 确保数值
                    for c in ["Open", "Close", "High", "Low", "Volume", "Amount"]:
                        if c in df_temp.columns:
                            df_temp[c] = pd.to_numeric(df_temp[c], errors="coerce")
                    
                    df_local = df_temp[["Date", "Open", "Close", "High", "Low", "Volume", "Amount"]].copy()
                    df_local = df_local.dropna(subset=["Date", "Close"])
                    df_local = df_local.sort_values("Date").reset_index(drop=True)
                    st.session_state["last_30m_path"] = f"Local CSV: {os.path.basename(fpath)}"
                    st.session_state["last_30m_cols"] = list(df_local.columns)
            except Exception as e:
                print(f"Read local csv failed: {e}")
    except Exception as e:
        pass

    # 如果没有找到 CSV，尝试旧的 Excel 逻辑
    if df_local.empty:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 定义可能的模式
            patterns = [
                f"Table{code}-*.xlsx",           # 新格式
                f"{code}ETF_30*.xlsx",           # 旧格式 (e.g. 159915ETF_30（2013-2019）.xlsx)
                f"*{code}*30*.xlsx"              # 通用匹配
            ]
            
            all_files = []
            for p in patterns:
                found = glob.glob(os.path.join(base_dir, p))
                all_files.extend(found)
                
            # 去重
            all_files = sorted(list(set(all_files)))
            
            # 排除临时文件
            all_files = [f for f in all_files if not os.path.basename(f).startswith("~$")]
            
            if all_files:
                dfs = []
                for fpath in all_files:
                    try:
                        # 读取
                        temp = pd.read_excel(fpath, engine="openpyxl")
                        
                        # 清洗列名
                        rename_dict = {}
                        for col in temp.columns:
                            name = str(col).strip()
                            low = name.lower().replace(" ", "")
                            if "date" in low or "time" in low or "datetime" in low or "日期" in name or "时间" in name:
                                rename_dict[col] = "Date"
                            elif ("收" in name and "盘" in name) or ("close" in low):
                                rename_dict[col] = "Close"
                            elif ("开" in name and "盘" in name) or ("open" in low):
                                rename_dict[col] = "Open"
                            elif ("高" in name and ("价" in name or True)) or ("high" in low):
                                rename_dict[col] = "High"
                            elif ("低" in name and ("价" in name or True)) or ("low" in low):
                                rename_dict[col] = "Low"
                            elif ("量" in name) or ("手" in name) or ("vol" in low):
                                rename_dict[col] = "Volume"
                            elif ("额" in name) or ("金额" in name) or ("amount" in low) or ("value" in low):
                                rename_dict[col] = "Amount"
                        
                        temp = temp.rename(columns=rename_dict)
                        
                        # 兜底：若第一列未识别为Date且包含时间信息
                        if "Date" not in temp.columns and not temp.empty:
                            temp = temp.rename(columns={temp.columns[0]: "Date"})
                            
                        # 清洗Date
                        if "Date" in temp.columns:
                            if pd.api.types.is_numeric_dtype(temp["Date"]):
                                base = pd.Timestamp("1899-12-30")
                                temp["Date"] = base + pd.to_timedelta(temp["Date"], unit="D")
                            else:
                                # 移除中文星期等
                                s = temp["Date"].astype(str).str.replace(r"[,\，][一二三四五六日天]$", "", regex=True)
                                temp["Date"] = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                                
                            # 清洗数值
                            for c in ["Open", "Close", "High", "Low", "Volume", "Amount"]:
                                if c in temp.columns:
                                    temp[c] = pd.to_numeric(temp[c], errors="coerce")
                            
                            keep_cols = ["Date"] + [c for c in ["Open", "Close", "High", "Low", "Volume", "Amount"] if c in temp.columns]
                            temp = temp[keep_cols].dropna(subset=["Date"])
                            
                            # 检查是否包含时间信息（不仅仅是00:00:00）
                            # 如果是30分钟数据，应该有多个不同的时间点
                            if len(temp) > 1:
                                time_std = temp["Date"].dt.hour.std() + temp["Date"].dt.minute.std()
                                if time_std == 0:
                                    # 所有时间都相同（例如都是00:00），视为日线数据误入
                                    # print(f"Skipping {fpath}: appears to be daily data (no time variation)")
                                    continue
                                    
                            dfs.append(temp)
                    except Exception as e:
                        print(f"Failed to load {fpath}: {e}")
                
                if dfs:
                    df_local = pd.concat(dfs, ignore_index=True)
                    df_local = df_local.drop_duplicates(subset=["Date"], keep="last")
                    df_local = df_local.sort_values("Date").reset_index(drop=True)
                    st.session_state["last_30m_path"] = f"Merged {len(all_files)} files"
                    st.session_state["last_30m_cols"] = list(df_local.columns)
                    
        except Exception as e:
            st.warning(f"本地Excel数据加载异常: {e}")
        
    # 2. 加载在线API数据 (优先使用 Baostock，失败则使用 Akshare)
    df_api = pd.DataFrame()
    api_source = "None"
    
    # --- 尝试 Baostock ---
    try:
        import baostock as bs
        
        # 转换代码格式：sh.600000 或 sz.000001
        def get_baostock_code(c):
            s = str(c)
            if s.startswith("6"): return f"sh.{s}"
            if s.startswith(("0", "3")): return f"sz.{s}"
            # 创业板ETF 159915 -> sz.159915
            if s.startswith("1"): return f"sz.{s}" 
            if s.startswith("5"): return f"sh.{s}"
            return f"sz.{s}" # 默认
            
        bs_code = get_baostock_code(code)
        
        # 登录
        bs.login()
        
        # 设定时间范围：过去3年到今天
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365*3)
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            frequency="30",
            adjustflag="2" # 前复权
        )
        
        if rs.error_code == '0':
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df_bs = pd.DataFrame(data_list, columns=rs.fields)
                
                # 处理时间格式：Baostock time 是 YYYYMMDDHHMMSSsss
                # date 是 YYYY-MM-DD
                # 我们主要需要合并成 datetime
                # 示例: time='20240228100000000' -> 2024-02-28 10:00:00
                # 或者直接解析 time 列
                
                df_bs['Date'] = pd.to_datetime(df_bs['time'], format='%Y%m%d%H%M%S%f', errors='coerce')
                
                # 重命名列
                rename_dict = {
                    'open': 'Open', 'high': 'High', 'low': 'Low', 
                    'close': 'Close', 'volume': 'Volume', 'amount': 'Amount'
                }
                df_bs = df_bs.rename(columns=rename_dict)
                
                # 转换数值
                for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']:
                    df_bs[c] = pd.to_numeric(df_bs[c], errors='coerce')
                
                # 筛选列
                keep_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
                df_api = df_bs[keep_cols].dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
                api_source = "Baostock"
                st.session_state["last_30m_api_status"] = f"Baostock获取成功: {len(df_api)}条"
        
        bs.logout()
        
    except Exception as e:
        # print(f"Baostock failed: {e}")
        try:
            bs.logout()
        except:
            pass
            
    # --- 如果 Baostock 失败或为空，尝试 Akshare (东方财富/新浪) ---
    if df_api.empty:
        try:
            import akshare as ak
            def get_exchange_prefix(c):
                s = str(c)
                return "sz" if s.startswith(("0", "1", "2", "3")) else "sh"
            prefix = get_exchange_prefix(code)
            symbol = f"{prefix}{code}"
            
            # 尝试不同接口
            temp_df = None
            
            # 优先尝试东方财富 (更稳定)
            try:
                # stock_zh_a_hist_min_em 接口
                temp_df = ak.stock_zh_a_hist_min_em(symbol=code, period="30", adjust="qfq")
                if temp_df is not None and not temp_df.empty:
                     # 东方财富返回列名：时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额, ...
                     rename_dict = {
                         "时间": "Date", "开盘": "Open", "收盘": "Close", 
                         "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Amount"
                     }
                     temp_df = temp_df.rename(columns=rename_dict)
            except:
                pass

            if temp_df is None or temp_df.empty:
                try:
                    temp_df = ak.fund_etf_min_sina(symbol=symbol, period="30")
                except Exception:
                    pass
                
            if temp_df is None or temp_df.empty:
                try:
                    temp_df = ak.stock_zh_a_minute(symbol=symbol, period="30")
                except Exception:
                    temp_df = None
                    
            if temp_df is not None and not temp_df.empty:
                # 清洗API数据列名
                lower_map = {str(col).lower(): col for col in temp_df.columns}
                rename_dict = {}
                if "day" in lower_map: rename_dict[lower_map["day"]] = "Date"
                if "date" in lower_map: rename_dict[lower_map["date"]] = "Date"
                if "time" in lower_map: rename_dict[lower_map["time"]] = "Date"
                if "open" in lower_map: rename_dict[lower_map["open"]] = "Open"
                if "close" in lower_map: rename_dict[lower_map["close"]] = "Close"
                if "high" in lower_map: rename_dict[lower_map["high"]] = "High"
                if "low" in lower_map: rename_dict[lower_map["low"]] = "Low"
                if "volume" in lower_map: rename_dict[lower_map["volume"]] = "Volume"
                if "amount" in lower_map: rename_dict[lower_map["amount"]] = "Amount"
                
                # 中文列名兜底
                cn_map = {"开盘": "Open", "收盘": "Close", "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Amount"}
                for cn, en in cn_map.items():
                    if cn in temp_df.columns:
                        rename_dict[cn] = en
                        
                temp_df = temp_df.rename(columns=rename_dict)
                
                if "Date" in temp_df.columns:
                    temp_df["Date"] = pd.to_datetime(temp_df["Date"], errors="coerce")
                    for c in ["Open", "Close", "High", "Low", "Volume", "Amount"]:
                        if c in temp_df.columns:
                            temp_df[c] = pd.to_numeric(temp_df[c], errors="coerce")
                            
                    keep_cols = ["Date"] + [c for c in ["Open", "Close", "High", "Low", "Volume", "Amount"] if c in temp_df.columns]
                    df_api = temp_df[keep_cols].dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
                    api_source = "Akshare"
                    st.session_state["last_30m_api_status"] = f"Akshare获取成功: {len(df_api)}条"
                else:
                     st.session_state["last_30m_api_status"] = "API数据缺少日期列"
            else:
                if api_source == "None":
                    st.session_state["last_30m_api_status"] = "API返回为空"
                
        except Exception as e:
            if api_source == "None":
                st.session_state["last_30m_api_status"] = f"API异常: {e}"
            # 不中断，继续使用本地数据

    # 3. 合并数据
    if df_local.empty and df_api.empty:
        st.error("无法获取30分钟数据（本地文件与API均失效）")
        return None
        
    if df_local.empty:
        df_final = df_api
    elif df_api.empty:
        df_final = df_local
    else:
        # 合并并去重
        df_final = pd.concat([df_local, df_api], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=["Date"], keep="last")
        df_final = df_final.sort_values("Date").reset_index(drop=True)

    # 过滤掉00:00时间点的数据（日线数据），只保留真正的30分钟数据
    if not df_final.empty and "Date" in df_final.columns:
        df_final["Date"] = pd.to_datetime(df_final["Date"])
        # 保留小时不为0或分钟不为0的数据（即非00:00的数据）
        df_final = df_final[(df_final["Date"].dt.hour != 0) | (df_final["Date"].dt.minute != 0)]
        df_final = df_final.reset_index(drop=True)
    
    # 4. 自动保存更新后的数据到本地 (增量更新)
    if not df_api.empty and len(df_final) > len(df_local):
        try:
            start_str = df_final["Date"].iloc[0].strftime("%Y%m%d")
            end_str = df_final["Date"].iloc[-1].strftime("%Y%m%d")
            
            # 构建新文件名
            base_dir = os.path.dirname(os.path.abspath(__file__))
            new_filename = f"Table{code}-{start_str}-{end_str}.xlsx"
            new_path = os.path.join(base_dir, new_filename)
            
            # 如果新文件名与旧文件名不同，且旧文件存在，则准备替换
            # 注意：如果正在使用旧文件，直接覆盖通常没问题，但为了安全，我们可以保存为新文件
            # 这里简单策略：保存为新文件，若保存成功，更新 session_state
            
            # 只有当新路径与当前使用的路径不同，或者确实有新数据时才保存
            # (这里条件 len(df_final) > len(df_local) 已经保证了有新数据)
            
            df_final.to_excel(new_path, index=False, engine="openpyxl")
            st.session_state["data_saved"] = f"已更新本地数据: {new_filename}"
            
            # 尝试清理旧文件 (可选，避免文件堆积)
            if "last_30m_path" in st.session_state:
                old_path = st.session_state["last_30m_path"]
                if old_path and os.path.exists(old_path) and os.path.abspath(old_path) != os.path.abspath(new_path):
                    try:
                        # 检查旧文件是否符合我们的命名模式 TableCode-Start-End.xlsx，防止误删其他文件
                        if f"Table{code}-" in os.path.basename(old_path):
                            os.remove(old_path)
                            st.session_state["data_cleaned"] = f"已清理旧文件: {os.path.basename(old_path)}"
                    except Exception as e:
                        print(f"清理旧文件失败: {e}")
                        
            st.session_state["last_30m_path"] = new_path
            
        except Exception as e:
            st.warning(f"自动保存数据失败: {e}")

    st.session_state["data_source_info"] = f"本地:{len(df_local)}条 + API({api_source}):{len(df_api)}条 -> 最终:{len(df_final)}条"
    return df_final




# ============================================
# 技术指标计算
# ============================================

def calculate_indicators(df, ma_lengths=None):
    """计算技术指标"""
    # 均线（动态，根据所需长度补充计算）
    default_ma = [10, 20, 60]
    ma_lengths = sorted(set((ma_lengths or []) + default_ma))
    for n in ma_lengths:
        col = f"MA{n}"
        if col not in df.columns:
            df[col] = df['Close'].rolling(n).mean()
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ROC
    df['ROC5'] = df['Close'].pct_change(5)
    df['ROC20'] = df['Close'].pct_change(20)
    
    return df


def select_ma_by_atr(row, atr_threshold=1.5, hi_pair=(10, 20), lo_pair=(20, 60)):
    if pd.isna(row['ATR_Pct']):
        short, long = lo_pair
        return (f'MA{short}', f'MA{long}')
    if row['ATR_Pct'] > atr_threshold:
        short, long = hi_pair
        return (f'MA{short}', f'MA{long}')
    else:
        short, long = lo_pair
        return (f'MA{short}', f'MA{long}')


def generate_signal(df, rsi_low=40, rsi_high=70, atr_threshold=1.5, hi_pair=(10, 20), lo_pair=(20, 60)):
    df_valid = df.dropna(subset=["Close", "High", "Low"])
    if len(df_valid) < 60:
        return None, None, None
    
    last_row = df_valid.iloc[-1]
    prev_row = df_valid.iloc[-2] if len(df_valid) > 1 else last_row
    
    short_ma, long_ma = select_ma_by_atr(last_row, atr_threshold, hi_pair=hi_pair, lo_pair=lo_pair)
    short_ma_val = last_row[short_ma]
    long_ma_val = last_row[long_ma]
    
    price = last_row['Close']
    rsi = last_row['RSI']
    roc20 = last_row['ROC20']
    atr_pct = last_row['ATR_Pct']
    
    buy_cond = (price > short_ma_val) and (short_ma_val > long_ma_val)
    buy_filter = (rsi >= rsi_low) and (rsi <= rsi_high) and (roc20 > 0)
    
    rsi_upper = min(100, rsi_high + 10)
    rsi_lower = max(0, rsi_low - 10)
    sell_cond = (price < short_ma_val) or (short_ma_val < long_ma_val)
    sell_filter = (rsi > rsi_upper) or (rsi < rsi_lower)
    
    if buy_cond and buy_filter:
        signal = "买入"
        reason = f"价格>{short_ma}>{long_ma}，RSI={rsi:.1f}∈[{rsi_low},{rsi_high}]，ROC20>0"
    elif sell_cond or sell_filter:
        signal = "卖出"
        if rsi > rsi_upper:
            reason = f"RSI超买={rsi:.1f}>{rsi_upper}"
        elif rsi < rsi_lower:
            reason = f"RSI超卖={rsi:.1f}<{rsi_lower}"
        else:
            reason = f"价格<{short_ma}或均线死叉"
    else:
        signal = "等待"
        reason = "若已持仓：维持持仓；若空仓：继续观望"
    
    return signal, reason, {
        'price': price,
        'short_ma': short_ma,
        'short_ma_val': short_ma_val,
        'long_ma': long_ma,
        'long_ma_val': long_ma_val,
        'rsi': rsi,
        'roc20': roc20,
        'atr_pct': atr_pct
    }


def solve_rsi_price(prev_close, sum_gain, sum_loss, n, target_rsi):
    """
    求解达到目标RSI所需的价格
    RSI = 100 - 100 / (1 + RS)
    RS = AvgGain / AvgLoss
    Simple Moving Average算法:
    AvgGain = (Sum_Past_Gain + Curr_Gain) / N
    AvgLoss = (Sum_Past_Loss + Curr_Loss) / N
    """
    if target_rsi == 100: return float('inf')
    if target_rsi == 0: return 0.0
    
    target_rs = target_rsi / (100 - target_rsi)
    
    # Case 1: 假设上涨 (P > PrevClose) -> Gain = P - Prev, Loss = 0
    # target_rs = (SumGain + P - Prev) / SumLoss
    # P = target_rs * SumLoss - SumGain + Prev
    p_up = target_rs * sum_loss - sum_gain + prev_close
    if p_up > prev_close:
        return p_up
        
    # Case 2: 假设下跌 (P < PrevClose) -> Gain = 0, Loss = Prev - P
    # target_rs = SumGain / (SumLoss + Prev - P)
    # SumLoss + Prev - P = SumGain / target_rs
    # P = SumLoss + Prev - SumGain / target_rs
    if target_rs != 0:
        p_down = sum_loss + prev_close - sum_gain / target_rs
        if p_down < prev_close:
            return p_down
            
    # Case 3: P = PrevClose
    # RS = SumGain / SumLoss
    # Check if this RS matches target (Unlikely to match exactly, but for completeness)
    return prev_close

def calculate_forecast(df, hi_pair=(10, 20), lo_pair=(20, 60), atr_threshold=1.5, rsi_low=40, rsi_high=70):
    """
    计算关键价位预告
    基于当前已完成的K线（df[:-1]），预测当前K线（df[-1]）收盘价达到多少时会触发信号变化
    """
    if len(df) < 60:
        return None
        
    # 假设当前行是正在进行的行，我们需要它的前一行作为基准
    # 如果盘中实时更新，df[-1]是当前bar
    # 我们需要计算基于 df[:-1] 的历史数据，来求解 df[-1]['Close'] 应该是多少
    
    # 1. 确定使用的均线参数
    # 这里有个循环依赖：ATR决定均线选择，但ATR也受当前收盘价影响。
    # 为简化，我们假设ATR状态不发生突变，使用当前的ATR值来锁定均线对。
    last_row = df.iloc[-1]
    short_ma_name, long_ma_name = select_ma_by_atr(last_row, atr_threshold, hi_pair, lo_pair)
    n_short = int(short_ma_name.replace("MA", ""))
    n_long = int(long_ma_name.replace("MA", ""))
    
    # 2. 准备历史数据 (排除当前bar，获取前N-1个数据)
    prev_close = df['Close'].iloc[-2]
    
    # MA Sums
    # MA_N = (Sum(Close[i-N+1 : i-1]) + P) / N
    # Need Sum of last N-1 closes excluding current
    closes = df['Close'].values
    # data includes current bar at -1. 
    # History for MA calculation:
    sum_short_prev = np.sum(closes[-n_short:-1]) if len(closes) >= n_short else np.sum(closes[:-1])
    sum_long_prev = np.sum(closes[-n_long:-1]) if len(closes) >= n_long else np.sum(closes[:-1])
    
    # RSI Sums (Simple Moving Average for RSI as per logic)
    # Gain/Loss calculation
    # We need the sum of the past N-1 gains/losses.
    # The 'calculate_indicators' computes rolling mean.
    # To reverse it: Sum_14 = Mean_14 * 14
    # The 'df' passed in already has 'RSI' computed, but that's the FINAL value.
    # We need the components. It's safer to re-calculate the sums from raw price changes.
    delta = df['Close'].diff()
    gains = delta.where(delta > 0, 0).values
    losses = (-delta.where(delta < 0, 0)).values
    
    # rolling(14) includes current.
    # sum_gain_14 = sum(gains[-14:])
    # We want sum of past 13 gains.
    rsi_n = 14
    sum_gain_prev = np.sum(gains[-rsi_n:-1]) 
    sum_loss_prev = np.sum(losses[-rsi_n:-1])
    
    # 3. Calculate Critical Prices
    
    # A. Price vs Short MA Cross
    # P > MA_S => P > (Sum_S + P)/N => P > Sum_S / (N-1)
    p_cross_short = sum_short_prev / (n_short - 1)
    
    # B. Short MA vs Long MA Cross
    # MA_S > MA_L => (Sum_S + P)/Ns > (Sum_L + P)/Nl
    # P > (Ns*Sum_L - Nl*Sum_S) / (Nl - Ns)
    p_ma_gold = (n_short * sum_long_prev - n_long * sum_short_prev) / (n_long - n_short)
    
    # C. RSI Targets
    p_rsi_low = solve_rsi_price(prev_close, sum_gain_prev, sum_loss_prev, rsi_n, rsi_low)
    p_rsi_high = solve_rsi_price(prev_close, sum_gain_prev, sum_loss_prev, rsi_n, rsi_high)
    p_rsi_upper = solve_rsi_price(prev_close, sum_gain_prev, sum_loss_prev, rsi_n, min(100, rsi_high + 10))
    p_rsi_lower = solve_rsi_price(prev_close, sum_gain_prev, sum_loss_prev, rsi_n, max(0, rsi_low - 10))
    
    return {
        "n_short": n_short,
        "n_long": n_long,
        "p_cross_short": p_cross_short,
        "p_ma_gold": p_ma_gold,
        "p_rsi_low": p_rsi_low,
        "p_rsi_high": p_rsi_high,
        "p_rsi_upper": p_rsi_upper,
        "p_rsi_lower": p_rsi_lower,
        "current_price": last_row['Close']
    }

def compute_signal_points(df, rsi_low=40, rsi_high=70, atr_threshold=1.5, hi_pair=(10, 20), lo_pair=(20, 60)):
    needed_ma = list(set([hi_pair[0], hi_pair[1], lo_pair[0], lo_pair[1]]))
    df_calc = calculate_indicators(df.copy(), ma_lengths=needed_ma)
    buys_x, buys_y, sells_x, sells_y = [], [], [], []
    in_pos = False
    for i in range(60, len(df_calc)):
        row = df_calc.iloc[i]
        short_ma, long_ma = select_ma_by_atr(row, atr_threshold, hi_pair=hi_pair, lo_pair=lo_pair)
        buy = (row["Close"] > row[short_ma]) and (row[short_ma] > row[long_ma]) and (row["RSI"] >= rsi_low) and (row["RSI"] <= rsi_high) and (row["ROC20"] > 0)
        rsi_upper = min(100, rsi_high + 10)
        rsi_lower = max(0, rsi_low - 10)
        sell = (row["Close"] < row[short_ma]) or (row[short_ma] < row[long_ma]) or (row["RSI"] > rsi_upper) or (row["RSI"] < rsi_lower)
        if not in_pos and buy:
            in_pos = True
            buys_x.append(row["Date"])
            buys_y.append(row["Close"])
        elif in_pos and sell:
            in_pos = False
            sells_x.append(row["Date"])
            sells_y.append(row["Close"])
    return buys_x, buys_y, sells_x, sells_y

# ============================================
# 回测模块
# ============================================

def backtest_strategy(df, rsi_low=40, rsi_high=70, atr_threshold=1.5, hi_pair=(10, 20), lo_pair=(20, 60)):
    """策略回测（含ETF手续费：佣金万1.2双向，免印花税和过户费）"""
    needed_ma = list(set([hi_pair[0], hi_pair[1], lo_pair[0], lo_pair[1]]))
    df = calculate_indicators(df, ma_lengths=needed_ma)

    # ETF交易费用：佣金万1.2（双向），免印花税和过户费
    commission_rate = 0.00012  # 0.012%

    in_position = False
    entry_price = 0
    entry_cost = 0  # 实际买入成本（含手续费）
    trades = []
    position_returns = np.zeros(len(df))

    for i in range(60, len(df)):
        current_price = df['Close'].iloc[i]

        short_ma, long_ma = select_ma_by_atr(df.iloc[i], atr_threshold, hi_pair=hi_pair, lo_pair=lo_pair)
        short_ma_val = df.iloc[i][short_ma]
        long_ma_val = df.iloc[i][long_ma]
        rsi = df.iloc[i]['RSI']
        roc20 = df.iloc[i]['ROC20']

        buy_cond = (current_price > short_ma_val) and (short_ma_val > long_ma_val)
        buy_filter = (rsi >= rsi_low) and (rsi <= rsi_high) and (roc20 > 0)

        sell_cond = (current_price < short_ma_val) or (short_ma_val < long_ma_val)
        rsi_upper = min(100, rsi_high + 10)
        rsi_lower = max(0, rsi_low - 10)
        sell_filter = (rsi > rsi_upper) or (rsi < rsi_lower)

        if not in_position and buy_cond and buy_filter:
            in_position = True
            entry_price = current_price
            # 买入成本 = 价格 * (1 + 佣金)
            entry_cost = current_price * (1 + commission_rate)
            entry_idx = i
        elif in_position:
            position_returns[i] = df['Close'].pct_change().iloc[i]

            if sell_cond or sell_filter:
                # 卖出收入 = 价格 * (1 - 佣金)
                exit_revenue = current_price * (1 - commission_rate)
                # 实际收益 = (卖出收入 - 买入成本) / 买入成本
                pnl = (exit_revenue - entry_cost) / entry_cost
                trades.append({
                    'entry_date': df['Date'].iloc[entry_idx],
                    'entry_price': entry_price,
                    'entry_cost': entry_cost,  # 含手续费的实际成本
                    'exit_date': df['Date'].iloc[i],
                    'exit_price': current_price,
                    'exit_revenue': exit_revenue,  # 扣手续费后的实际收入
                    'pnl': pnl,
                    'commission': entry_price * commission_rate + current_price * commission_rate  # 总手续费
                })
                in_position = False

    # 强制平仓（最后一天）
    if in_position:
        last_price = df['Close'].iloc[-1]
        exit_revenue = last_price * (1 - commission_rate)
        pnl = (exit_revenue - entry_cost) / entry_cost
        trades.append({
            'entry_date': df['Date'].iloc[entry_idx],
            'entry_price': entry_price,
            'entry_cost': entry_cost,
            'exit_date': df['Date'].iloc[-1],
            'exit_price': last_price,
            'exit_revenue': exit_revenue,
            'pnl': pnl,
            'commission': entry_price * commission_rate + last_price * commission_rate
        })

    # 计算收益（买入持有也扣除手续费以便公平对比）
    df['StrategyReturn'] = position_returns
    # 买入持有：买入和卖出各扣一次佣金
    df['BuyHoldReturn'] = df['Close'].pct_change().fillna(0) * (1 - 2 * commission_rate)

    cumret_bh = (1 + df['BuyHoldReturn']).cumprod()
    cumret_strat = (1 + df['StrategyReturn']).cumprod()

    return df, trades, cumret_bh, cumret_strat


# ============================================
# 界面显示
# ============================================

st.title("📈 AI 智能量化决策系统")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 设置")
    
    # 股票代码输入
    # 强制将输入转换为字符串，防止 Streamlit 将纯数字代码识别为整数进行处理（这可能导致前导0丢失或其他奇怪的格式化问题）
    # 但 text_input 返回的本就是字符串。问题可能出在 value 参数的来源。
    # 检查 st.session_state.get("target_code") 是否被存为了整数。
    
    current_target = st.session_state.get("target_code", DEFAULT_CODE)
    if not isinstance(current_target, str):
        current_target = str(current_target)
        
    # --- 拼音搜索功能支持 ---
    search_enabled = st.checkbox("启用拼音/名称搜索", value=st.session_state.get("enable_search", True), key="enable_search")
    
    new_code = current_target
    
    if search_enabled:
        stock_options = []
        try:
            import stock_utils
            @st.cache_data(ttl=3600*24, show_spinner="正在加载股票列表...")
            def load_options():
                return stock_utils.get_stock_options()
            stock_options = load_options()
        except Exception as e:
            st.warning(f"无法加载股票列表: {e}")
            
        if stock_options:
            # 查找当前代码在列表中的位置
            current_idx = 0
            options_labels = [x['label'] for x in stock_options]
            
            # 尝试匹配当前代码
            # 为了性能，我们可以先检查是否已经匹配（如果 session_state 中有记录）
            # 这里简单遍历，5000个元素还是很快的
            match_found = False
            for i, opt in enumerate(stock_options):
                if opt['code'] == current_target:
                    current_idx = i
                    match_found = True
                    break
            
            # 如果没找到完全匹配的，但 current_target 不为空，可能是在列表中不存在
            # 这时 selectbox 会默认显示第一个，或者我们可以添加一个临时选项
            if not match_found and current_target:
                # 添加临时选项以保持显示
                temp_label = f"{current_target} (当前未知)"
                options_labels.insert(0, temp_label)
                current_idx = 0
            
            selected = st.selectbox("搜索股票/基金 (输入拼音首字母)", options=options_labels, index=current_idx)
            if selected:
                new_code = selected.split(" ")[0]
        else:
            st.info("股票列表加载中或为空，请稍候...")
            new_code = st.text_input("股票/ETF代码", value=current_target)
    else:
        new_code = st.text_input("股票/ETF代码", value=current_target)
    
    # 获取实时名称
    rt_info = get_realtime_price(new_code)
    if rt_info:
        st.caption(f"📌 {rt_info['name']} ({new_code})")
    else:
        st.caption(f"📌 {new_code}")
    
    # 检查是否更换了代码
    # 处理输入值：去除空白字符
    new_code = new_code.strip()
    
    if new_code != st.session_state.get("target_code"):
        st.session_state["target_code"] = new_code
        ETF_CODE = new_code
        
        # 同步更新预测模块的股票代码
        st.session_state["pred_symbol"] = new_code
        # 标记需要自动运行预测 (但仅当用户真正切换到预测 tab 时才运行，或者我们可以静默运行)
        # 为了不阻塞主界面，我们只设置标记，当用户点击 Tab4 时，stock_prediction_module 会读取此标记并自动运行
        st.session_state["auto_run_prediction"] = True
        
        # 尝试立即在后台启动预测任务（非阻塞）
        # 利用 Streamlit 的执行机制，在这里调用一个不输出到 UI 的函数
        if spm and spm.torch is not None:
            # 注意：这实际上还是会阻塞当前脚本执行，除非用 threading
            # 为了真正的用户体验，我们最好还是让它在后台线程跑，或者在主页面渲染完之后跑
            # 简单起见，我们先保持标记位逻辑，或者尝试直接调用（如果用户能接受多等几秒）
            # 用户要求“系统生成完实时监控及回测分析及交易记录代码后，要立即自动计算”
            # 所以我们应该把这个调用放在页面底部
            pass
        
        # 优化：检查是否真的需要重新下载和优化
        # 1. 检查是否存在已保存的策略
        saved_s = load_stock_strategy(new_code)
        need_optimization = True
        
        # 2. 获取最新数据的日期
        latest_date_str = ""
        try:
            # 尝试获取最近一天的数据日期，避免只看策略生成时间
            # 如果本地有数据文件，读取文件最后一行
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_files = glob.glob(os.path.join(base_dir, f"Daily_{new_code}_*.xlsx"))
            if local_files:
                local_files.sort(key=os.path.getmtime, reverse=True)
                # 从文件名解析日期? Daily_159915_20260222.xlsx
                fname = os.path.basename(local_files[0])
                if "_" in fname:
                    parts = fname.split("_")
                    if len(parts) >= 3:
                        latest_date_str = parts[2].replace(".xlsx", "")
        except:
            pass

        if saved_s and "updated_at" in saved_s:
            try:
                last_update = datetime.strptime(saved_s["updated_at"], "%Y-%m-%d %H:%M:%S")
                # 策略更新时间
                update_date_str = last_update.strftime("%Y%m%d")
                
                # 判定逻辑优化：
                # 1. 如果策略是今天生成的 -> 肯定最新，跳过
                # 2. 如果策略生成日期 >= 本地数据文件的日期 -> 说明策略是基于最新数据生成的，跳过
                # 3. 否则 -> 需要更新
                
                is_today = last_update.date() == datetime.now().date()
                is_newer_than_data = (latest_date_str and update_date_str >= latest_date_str)
                
                if (is_today or is_newer_than_data) and "daily" in saved_s and "30min" in saved_s:
                    need_optimization = False
                    st.success(f"检测到 {new_code} 的策略已是最新 ({saved_s['updated_at']})，跳过重复计算。")
            except:
                pass
        
        # 如果不需要优化，我们仍然要确保 pred_symbol 更新并触发自动预测（如果还没有结果）
        if not need_optimization:
             # 如果之前没有触发过自动预测，或者切换了股票，这里再次确认触发
             # 但注意 auto_run_prediction 已经在上面设置了 True
             # 这里主要是为了防止“如果不需要优化就不rerun”导致的预测未触发
             # 实际上我们在下面有一行 safe_rerun()，所以这里不需要额外操作
             pass

        if need_optimization:
            # 自动触发流程：下载数据 -> 优化日线 -> 优化30分钟 -> 保存 -> 加载
            with st.spinner(f"正在初始化 {new_code}... 下载数据并自动优化策略..."):
                status_text = st.empty()
                
                # 1. 下载并优化日线
                status_text.text("正在优化日线策略...")
                df_daily = get_history_data_daily(new_code, days=5000)
                best_daily = None
                if df_daily is not None and len(df_daily) > 200:
                    best_daily = optimize_strategy_for_stock(df_daily, "日线")
                
                # 2. 下载并优化30分钟
                status_text.text("正在优化30分钟策略...")
                df_30m = get_history_data_30min(new_code)
                best_30m = None
                if df_30m is not None and len(df_30m) > 200:
                    best_30m = optimize_strategy_for_stock(df_30m, "30分钟")
                
                # 3. 保存策略
                strategy_data = {
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "daily": best_daily,
                    "30min": best_30m
                }
                
                # 读取旧数据以防覆盖其他股票
                all_strategies = {}
                if os.path.exists(STRATEGY_FILE):
                    try:
                        with open(STRATEGY_FILE, "r", encoding="utf-8") as f:
                            all_strategies = pyjson.load(f)
                    except:
                        pass
                
                all_strategies[new_code] = strategy_data
                with open(STRATEGY_FILE, "w", encoding="utf-8") as f:
                    pyjson.dump(all_strategies, f, ensure_ascii=False, indent=2)
                
                status_text.text("策略优化完成！正在加载...")
                
                # 4. 自动加载当前周期的策略
                # 强制刷新以应用新策略
                # 在刷新前，确保 pending_params 设置为当前周期的最优参数
                # 这样当页面重新加载时，会直接应用正确的参数
                target_p_load = None
                current_tf_now = st.session_state.get("timeframe", "日线")
                if current_tf_now == "30分钟":
                    target_p_load = best_30m
                else:
                    target_p_load = best_daily
                
                if target_p_load:
                    st.session_state["pending_params"] = {
                        "atr_threshold": float(target_p_load["atr_threshold"]),
                        "rsi_low": int(target_p_load["rsi_low"]),
                        "rsi_high": int(target_p_load["rsi_high"]),
                        "hi_short": int(target_p_load["hi_short"]),
                        "hi_long": int(target_p_load["hi_long"]),
                        "lo_short": int(target_p_load["lo_short"]),
                        "lo_long": int(target_p_load["lo_long"]),
                    }
                    # 设置标识，防止后面再次加载默认值
                    st.session_state["last_loaded_strategy_key"] = f"strategy_loaded_{new_code}_{current_tf_now}"
                
                safe_rerun()
        else:
            # 不需要优化，但也需要刷新以加载已有策略
            # 清除之前的 pending_params 以防冲突？不，load逻辑在后面。
            # 直接rerun让后面的逻辑接管加载
            safe_rerun()
            
    else:
        ETF_CODE = new_code
    
    # 在渲染任何带 key 的控件之前，优先应用待加载的参数
    if "pending_params" in st.session_state and st.session_state["pending_params"]:
        p = st.session_state.pop("pending_params")
        st.session_state["atr_threshold"] = float(p.get("atr_threshold", 1.5))
        st.session_state["rsi_low"] = int(p.get("rsi_low", 40))
        st.session_state["rsi_high"] = int(p.get("rsi_high", 70))
        st.session_state["hi_short"] = int(p.get("hi_short", 10))
        st.session_state["hi_long"] = int(p.get("hi_long", 20))
        st.session_state["lo_short"] = int(p.get("lo_short", 20))
        st.session_state["lo_long"] = int(p.get("lo_long", 60))
    
    # 获取当前实际生效的数据源状态（如果有缓存）
    current_source_index = 0
    ds_options = ["akshare (免费)", "新浪 (实时)", "Baostock (证券宝)"]
    
    if "last_30m_api_status" in st.session_state:
        status = st.session_state["last_30m_api_status"]
        if "Baostock" in status:
            current_source_index = 2
        elif "Akshare" in status:
            current_source_index = 0
            
    # 如果用户没有显式选择过，我们尝试根据状态自动设定默认值
    # 但 selectbox 的 value 是根据 key 绑定的，如果用 index 控制显示
    
    # 简化处理：显示当前实际数据源状态
    if st.session_state.get("timeframe") == "30分钟" and "last_30m_api_status" in st.session_state:
        st.info(f"当前数据源: {st.session_state['last_30m_api_status']}")

    data_source = st.selectbox(
        "数据源接口 (首选)",
        ["akshare (免费)", "新浪 (实时)", "Baostock (证券宝)"],
        index=2, # 默认首选 Baostock
        help="系统会自动尝试首选源，失败后自动切换备用源"
    )
    # 数据周期选择
    timeframe = st.selectbox(
        "数据周期",
        ["日线", "30分钟"],
        index=0,
    )
    
    auto_refresh = st.checkbox("自动刷新当前价格 (10秒)", value=True, key="auto_refresh")
    
    # 自动加载策略逻辑
    # 优先级：
    # 1. 如果刚切换了代码且有该代码的专属策略 -> 加载专属策略
    # 2. 如果没有专属策略 -> 加载默认方案 (Preset 1/2)
    
    # 检查是否有针对当前代码的专属策略
    stock_strategy = load_stock_strategy(ETF_CODE)
    
    # 构建当前代码+周期的唯一标识，防止重复加载
    current_strategy_key = f"strategy_loaded_{ETF_CODE}_{timeframe}"
    
    # 如果还没加载过这个组合的策略
    if st.session_state.get("last_loaded_strategy_key") != current_strategy_key:
        
        # 尝试加载专属策略
        loaded_custom = False
        if stock_strategy:
            target_p = None
            if timeframe == "日线" and "daily" in stock_strategy:
                target_p = stock_strategy["daily"]
            elif timeframe == "30分钟" and "30min" in stock_strategy:
                target_p = stock_strategy["30min"]
                
            if target_p:
                st.session_state["pending_params"] = {
                    "atr_threshold": float(target_p["atr_threshold"]),
                    "rsi_low": int(target_p["rsi_low"]),
                    "rsi_high": int(target_p["rsi_high"]),
                    "hi_short": int(target_p["hi_short"]),
                    "hi_long": int(target_p["hi_long"]),
                    "lo_short": int(target_p["lo_short"]),
                    "lo_long": int(target_p["lo_long"]),
                }
                st.session_state["last_loaded_strategy_key"] = current_strategy_key
                loaded_custom = True
                safe_rerun()
        
        # 如果没有专属策略，则回退到默认 Preset 逻辑
        if not loaded_custom:
            # 尝试现场自动优化 (兜底逻辑)
            success_opt = False
            # 仅当数据足够时才尝试优化
            # 避免重复尝试：如果之前尝试过但失败了（数据不足），这里可能会陷入死循环？
            # 我们可以加一个标记，或者仅依靠 loaded_custom 判断
            
            # 检查当天是否已生成过策略
            strategy_cache_key = f"strategy_generated_{ETF_CODE}_{timeframe}"

            if is_cache_valid(strategy_cache_key):
                # 使用当天缓存的策略
                cached_strategy = get_cache(strategy_cache_key)
                if cached_strategy:
                    st.info(f"📦 使用今日已生成的{timeframe}策略 ({ETF_CODE})")
                    best_p = cached_strategy
                    df_opt = None  # 不需要重新获取数据
                else:
                    best_p = None
                    df_opt = None
            else:
                with st.spinner(f"未检测到 {ETF_CODE} 的 {timeframe} 专属策略，正在自动生成..."):
                    if timeframe == "日线":
                         df_opt = get_history_data_daily(ETF_CODE, days=5000)
                    else:
                         df_opt = get_history_data_30min(ETF_CODE)

                    if df_opt is not None and len(df_opt) > 200:
                        best_p = optimize_strategy_for_stock(df_opt, timeframe)
                        # 缓存当天生成的策略
                        if best_p:
                            set_cache(strategy_cache_key, best_p)
                    else:
                        best_p = None

                    if best_p:
                         # 增量保存策略
                         current_s = load_stock_strategy(ETF_CODE) or {}
                         current_s["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                         if timeframe == "日线":
                             current_s["daily"] = best_p
                         else:
                             current_s["30min"] = best_p
                         save_stock_strategy(ETF_CODE, current_s)
                         
                         # 应用新策略
                         st.session_state["pending_params"] = {
                            "atr_threshold": float(best_p["atr_threshold"]),
                            "rsi_low": int(best_p["rsi_low"]),
                            "rsi_high": int(best_p["rsi_high"]),
                            "hi_short": int(best_p["hi_short"]),
                            "hi_long": int(best_p["hi_long"]),
                            "lo_short": int(best_p["lo_short"]),
                            "lo_long": int(best_p["lo_long"]),
                         }
                         success_opt = True
            
            if success_opt:
                st.session_state["last_loaded_strategy_key"] = current_strategy_key
                st.success(f"已自动生成并加载 {timeframe} 专属策略")
                safe_rerun()

            # 如果自动优化失败（如数据不足），则加载默认 Preset
            desired_slot = 0 if timeframe == "日线" else 1
            # 确保presets已加载
            if "presets" not in st.session_state or not st.session_state["presets"]:
                 if os.path.exists(presets_path_early):
                    try:
                        with open(presets_path_early, "r", encoding="utf-8") as f:
                            st.session_state["presets"] = pyjson.load(f)
                    except:
                        pass
            
            p_list = st.session_state.get("presets")
            p = p_list[desired_slot] if p_list and len(p_list) > desired_slot else None
            
            if p:
                st.session_state["pending_params"] = p
                st.session_state["last_loaded_strategy_key"] = current_strategy_key
                safe_rerun()
            else:
                st.session_state["last_loaded_strategy_key"] = current_strategy_key

    st.header("📊 策略参数")
    atr_threshold = st.slider("ATR波动率阈值(%)", 1.0, 3.0, st.session_state.get("atr_threshold", 1.5), 0.1, key="atr_threshold")
    rsi_low = st.slider("RSI下限", 20, 50, st.session_state.get("rsi_low", 40), key="rsi_low")
    rsi_high = st.slider("RSI上限", 50, 90, st.session_state.get("rsi_high", 70), key="rsi_high")
    st.caption("均线参数（两种市场情况）")
    col_ma1, col_ma2 = st.columns(2)
    hi_short = col_ma1.number_input("高波动短均线", min_value=3, max_value=120, value=st.session_state.get("hi_short", 10), step=1, key="hi_short")
    hi_long = col_ma2.number_input("高波动长均线", min_value=5, max_value=240, value=st.session_state.get("hi_long", 20), step=1, key="hi_long")
    lo_short = col_ma1.number_input("低波动短均线", min_value=3, max_value=120, value=st.session_state.get("lo_short", 20), step=1, key="lo_short")
    lo_long = col_ma2.number_input("低波动长均线", min_value=5, max_value=240, value=st.session_state.get("lo_long", 60), step=1, key="lo_long")
    # 简单有效性约束
    if hi_short >= hi_long:
        hi_long = max(hi_short + 1, hi_long + 1)
    if lo_short >= lo_long:
        lo_long = max(lo_short + 1, lo_long + 1)
    hi_pair = (int(hi_short), int(hi_long))
    lo_pair = (int(lo_short), int(lo_long))
    
    st.divider()

    # 缓存控制按钮
    st.header("🔄 缓存控制")
    col_refresh1, col_refresh2 = st.columns(2)
    with col_refresh1:
        if st.button("🔄 刷新数据", help="强制重新下载今日数据", use_container_width=True):
            # 清除数据缓存
            clear_day_cache(f"daily_data_{ETF_CODE}_3000")
            clear_day_cache(f"30min_data_{ETF_CODE}")
            st.success("数据缓存已清除，将重新下载")
            safe_rerun()
    with col_refresh2:
        if st.button("🔄 刷新策略", help="强制重新生成今日策略", use_container_width=True):
            # 清除策略缓存
            clear_day_cache(f"strategy_generated_{ETF_CODE}_日线")
            clear_day_cache(f"strategy_generated_{ETF_CODE}_30分钟")
            st.success("策略缓存已清除，将重新生成")
            safe_rerun()
    st.caption("💡 同一天内数据和策略会自动缓存，点击按钮可强制刷新")

    st.divider()
    st.header("🧠 智能策略优化")
    
    # 加载已保存的策略
    saved_strategy = load_stock_strategy(ETF_CODE)
    if saved_strategy:
        st.info(f"已发现针对 {ETF_CODE} 的专属策略")
        
        # 兼容性处理：判断是新格式(分timeframe)还是旧格式(单层)
        display_ret = 0.0
        display_time = "未知"
        
        target_p = None
        if timeframe == "日线" and "daily" in saved_strategy:
            target_p = saved_strategy["daily"]
        elif timeframe == "30分钟" and "30min" in saved_strategy:
            target_p = saved_strategy["30min"]
        # 旧格式兜底 (如果 saved_strategy 本身就是参数字典)
        elif "atr_threshold" in saved_strategy:
            target_p = saved_strategy
            
        if target_p:
            display_ret = target_p.get("return", 0)
            display_time = saved_strategy.get("updated_at", "未知")
            st.caption(f"更新时间: {display_time} | 历史收益: {display_ret:.2f}%")
            
            if st.button("📥 加载专属策略", use_container_width=True):
                # 将策略转换为参数格式
                p_opt = {
                    "atr_threshold": float(target_p["atr_threshold"]),
                    "rsi_low": int(target_p["rsi_low"]),
                    "rsi_high": int(target_p["rsi_high"]),
                    "hi_short": int(target_p["hi_short"]),
                    "hi_long": int(target_p["hi_long"]),
                    "lo_short": int(target_p["lo_short"]),
                    "lo_long": int(target_p["lo_long"]),
                }
                apply_params(p_opt)
                st.success("已加载专属策略参数！")
                safe_rerun()
        else:
            st.caption(f"暂无 {timeframe} 专属策略")
    
    if st.button("🚀 重新生成/优化策略", use_container_width=True):
        with st.spinner("正在下载数据并进行全网格搜索优化...请稍候..."):
            # 1. 确保数据是最新的
            if timeframe == "日线":
                df_opt = get_history_data_daily(ETF_CODE, days=5000)
            else:
                df_opt = get_history_data_30min(ETF_CODE)
            
            if df_opt is not None and len(df_opt) > 200:
                # 2. 运行优化
                best_p = optimize_strategy_for_stock(df_opt, timeframe)
                if best_p:
                    # 3. 保存 (增量更新)
                    current_s = load_stock_strategy(ETF_CODE) or {}
                    current_s["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if timeframe == "日线":
                        current_s["daily"] = best_p
                    else:
                        current_s["30min"] = best_p
                    save_stock_strategy(ETF_CODE, current_s)
                    
                    st.success(f"优化完成！最佳收益率: {best_p['return']:.2f}%")
                    # 4. 自动加载
                    p_opt = {
                        "atr_threshold": float(best_p["atr_threshold"]),
                        "rsi_low": int(best_p["rsi_low"]),
                        "rsi_high": int(best_p["rsi_high"]),
                        "hi_short": int(best_p["hi_short"]),
                        "hi_long": int(best_p["hi_long"]),
                        "lo_short": int(best_p["lo_short"]),
                        "lo_long": int(best_p["lo_long"]),
                    }
                    apply_params(p_opt)
                    safe_rerun()
                else:
                    st.error("优化失败：未能找到正收益组合或数据不足")
            else:
                st.error("数据不足，无法进行有效优化")

    st.header("📈 收益对比")
    show_bh = st.checkbox("买入持有", value=True)
    show_strategy = st.checkbox("策略收益", value=True)

# 主界面
if st.session_state.get("auto_refresh", True) and st_autorefresh:
    st_autorefresh(interval=10_000, key="auto_refresh_counter")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 实时监控", "📈 回测分析", "📋 交易记录", "🔮 股价预测", "🤖 综合研判", "🎯 智能选股"])

# 获取实时数据 (在所有 Tab 之前获取，确保变量可用)
realtime_data = get_realtime_price(ETF_CODE)

with tab6:
    if ssm:
        ssm.render_selection_page()
    else:
        st.error("智能选股模块加载失败，请检查 stock_selection_module.py 是否存在。")

with tab5:
    if sam:
        # 准备数据
        current_price = realtime_data['close'] if realtime_data else 0
        
        # 准备信号数据
        # 优先从 session_state 获取
        sig_data = st.session_state.get("latest_signal_data", {})
        
        # 如果 session_state 为空，尝试现场计算（复用 tab1 的逻辑）
        if not sig_data:
            # 尝试计算双周期信号（日线 + 30分钟）以提供更丰富的信息给AI
            # 1. 日线信号
            df_d = get_history_data_daily(ETF_CODE, 6000)
            sig_d, reason_d = "未知", "无数据"
            if df_d is not None:
                # 获取日线参数 (方案一)
                p_d = st.session_state.get("presets", [None])[0] or {
                    "rsi_low": rsi_low, "rsi_high": rsi_high, "atr_threshold": atr_threshold,
                    "hi_short": hi_pair[0], "hi_long": hi_pair[1], "lo_short": lo_pair[0], "lo_long": lo_pair[1]
                }
                needed_ma = [p_d.get("hi_short", 10), p_d.get("hi_long", 20), p_d.get("lo_short", 20), p_d.get("lo_long", 60)]
                df_d = calculate_indicators(df_d, ma_lengths=needed_ma)
                s_d, r_d, _ = generate_signal(
                    df_d, 
                    rsi_low=int(p_d.get("rsi_low", 40)), 
                    rsi_high=int(p_d.get("rsi_high", 70)), 
                    atr_threshold=float(p_d.get("atr_threshold", 1.5)), 
                    hi_pair=(int(p_d.get("hi_short", 10)), int(p_d.get("hi_long", 20))), 
                    lo_pair=(int(p_d.get("lo_short", 20)), int(p_d.get("lo_long", 60)))
                )
                sig_d, reason_d = s_d, r_d

            # 2. 30分钟信号
            df_30 = get_history_data_30min(ETF_CODE)
            sig_30, reason_30 = "未知", "无数据"
            if df_30 is not None:
                # 获取30分钟参数 (方案二)
                p_30 = st.session_state.get("presets", [None, None])[1] or {
                    "rsi_low": rsi_low, "rsi_high": rsi_high, "atr_threshold": atr_threshold,
                    "hi_short": hi_pair[0], "hi_long": hi_pair[1], "lo_short": lo_pair[0], "lo_long": lo_pair[1]
                }
                needed_ma_30 = [p_30.get("hi_short", 10), p_30.get("hi_long", 20), p_30.get("lo_short", 20), p_30.get("lo_long", 60)]
                df_30 = calculate_indicators(df_30, ma_lengths=needed_ma_30)
                s_30, r_30, _ = generate_signal(
                    df_30, 
                    rsi_low=int(p_30.get("rsi_low", 40)), 
                    rsi_high=int(p_30.get("rsi_high", 70)), 
                    atr_threshold=float(p_30.get("atr_threshold", 1.5)), 
                    hi_pair=(int(p_30.get("hi_short", 10)), int(p_30.get("hi_long", 20))), 
                    lo_pair=(int(p_30.get("lo_short", 20)), int(p_30.get("lo_long", 60)))
                )
                sig_30, reason_30 = s_30, r_30

            sig_data = {
                "signal": sig_d, # 兼容旧逻辑
                "reason": reason_d,
                "daily_signal": {"signal": sig_d, "reason": reason_d},
                "min30_signal": {"signal": sig_30, "reason": reason_30}
            }
            # 存回 session 以备后用
            st.session_state["latest_signal_data"] = sig_data
        
        # 准备预测数据
        # 尝试从 session_state 获取缓存的预测结果
        pred_cache_key = f"pred_result_{ETF_CODE}_daily" # 默认取日线预测
        pred_df = st.session_state.get(pred_cache_key)
        
        if not sig_data:
            st.warning("请先切换到【实时监控】页签以生成最新的技术信号。")
        else:
            sam.render_analysis_page(
                symbol=ETF_CODE,
                current_price=current_price,
                signal_data=sig_data,
                pred_df=pred_df
            )
    else:
        st.error("无法加载分析模块。")

with tab4:
    # 股价预测模块
    if spm and spm.torch is not None:
        # 如果用户当前正在监控某个股票，我们可以尝试将其作为默认预测对象
        # 注意：这里 pred_symbol 已经通过 session_state 联动更新了
        if "pred_symbol" not in st.session_state:
            st.session_state["pred_symbol"] = ETF_CODE
        
        # 调用模块 (现在 stock_prediction_module 已经支持自动运行)
        # 只要 st.session_state["auto_run_prediction"] 为 True，
        # render_prediction_page 内部就会自动触发预测
        spm.render_prediction_page()
            
    else:
        st.warning("股价预测模块依赖库（如 torch）尚未加载成功，暂无法使用预测功能。请尝试重启 Streamlit 服务。")

with tab1:
    col1, col2, col3 = st.columns(3)
    
    # 获取实时数据
    # realtime = get_realtime_price(ETF_CODE) # 已经提前获取了
    
    # 动态显示数据源信息
    # 如果当前选择了30分钟，且已经获取了数据，我们可以从 session_state 中获取真正使用的数据源
    current_source_display = "akshare (免费)" # 默认
    if timeframe == "30分钟" and "last_30m_api_status" in st.session_state:
        status = st.session_state["last_30m_api_status"]
        if "Baostock" in status:
            current_source_display = "Baostock (证券宝)"
        elif "Akshare" in status:
            current_source_display = "Akshare (东方财富/新浪)"
    
    # 如果用户想手动切换，这个 selectbox 依然存在，但我们可以让它显示当前实际生效的源作为提示
    # 或者我们直接禁用它，因为现在是自动尝试
    # 为了保持界面一致性，我们可以更新 selectbox 的 options，并把当前生效的设为默认
    
    # 注意：selectbox 在 sidebar，而我们在 tab1 才确定了数据源。Streamlit 的渲染顺序是从上到下。
    # 所以要改变 sidebar 的显示，必须在 sidebar 渲染时就直到状态。
    # 但 get_history_data_30min 是在 tab1 调用的。
    # 解决方案：
    # 1. 将数据获取提前到 sidebar 之前（不推荐，影响性能）
    # 2. 接受 sidebar 显示滞后一次刷新（即第一次显示默认，刷新后显示真实）
    # 3. 或者我们在 sidebar 只是显示“当前策略使用的源”，而不是让用户选（因为代码里是自动 fallback）
    
    # 我们修改 sidebar 的 selectbox 为 info 显示，或者根据 session_state 动态调整 index
    pass

    
    if realtime_data:
        with col1:
            st.metric("当前价格", f"{realtime_data['close']:.3f}")
        with col2:
            base_price = realtime_data.get('pre_close', realtime_data['open'])
            change = realtime_data['close'] - base_price
            change_pct = change / base_price * 100
            st.metric("涨跌幅", f"{change_pct:+.2f}%", f"{change:+.3f}")
        with col3:
            st.metric("成交量", f"{realtime_data['volume']:,}手")
            
    # 在这里插入后台预测任务（静默运行）
    # 当页面刷新且有 auto_run_prediction 标记时
    if st.session_state.get("auto_run_prediction", False):
        if spm and spm.torch is not None:
            # 使用 spinner 提示用户正在后台计算
            # with st.spinner(f"正在后台计算 {ETF_CODE} 的股价预测..."): # 如果不想打扰用户可以去掉 spinner
            # 但用户要求“自动计算”，显示进度可能更好
            status_placeholder = st.empty()
            status_placeholder.caption(f"🤖 正在后台计算 {ETF_CODE} 的 AI 股价预测...")
            
            # 运行预测
            spm.run_prediction_background(ETF_CODE, "daily")
            
            status_placeholder.caption(f"✅ {ETF_CODE} 的 AI 股价预测已准备就绪 (请切换到“股价预测”标签页查看)")
            # 清除自动运行标记，避免重复计算
            st.session_state["auto_run_prediction"] = False
            # 这里的计算结果已经存入 session_state["pred_result_..."]
    
    # 技术指标和信号
    if 'timeframe' not in st.session_state or st.session_state.get('timeframe') != timeframe:
        st.session_state['timeframe'] = timeframe
    if timeframe == "30分钟":
        df = get_history_data_30min(ETF_CODE)
        if df is None or df.empty:
            st.error(f"30分钟数据为空。来源：{st.session_state.get('last_30m_path', '未知')} 列：{st.session_state.get('last_30m_cols', [])}")
            st.stop()
        else:
            min_date = df["Date"].min().strftime("%Y-%m-%d")
            max_date = df["Date"].max().strftime("%Y-%m-%d")
            st.sidebar.caption(f"30分钟数据范围: {min_date} 至 {max_date}")
    else:
        df = get_history_data_daily(ETF_CODE, 6000)
        if df is not None:
             min_date = df["Date"].min().strftime("%Y-%m-%d")
             max_date = df["Date"].max().strftime("%Y-%m-%d")
             st.sidebar.caption(f"日线数据范围: {min_date} 至 {max_date}")
    
    if df is not None:
        # 优先使用系统为该股票和周期生成的专属策略
        stock_strategy = load_stock_strategy(ETF_CODE)
        period_key = "30min" if timeframe == "30分钟" else "daily"

        if stock_strategy and period_key in stock_strategy:
            # 使用专属策略
            p_custom = stock_strategy[period_key]
            rsi_low_cur = int(p_custom.get("rsi_low", rsi_low))
            rsi_high_cur = int(p_custom.get("rsi_high", rsi_high))
            atr_cur = float(p_custom.get("atr_threshold", atr_threshold))
            hi_cur = (int(p_custom.get("hi_short", hi_pair[0])), int(p_custom.get("hi_long", hi_pair[1])))
            lo_cur = (int(p_custom.get("lo_short", lo_pair[0])), int(p_custom.get("lo_long", lo_pair[1])))
            st.info(f"{timeframe}策略使用专属优化方案 (收益率: {p_custom.get('return', 0):.1f}%)")
        else:
            # 回退到通用方案
            if timeframe == "30分钟":
                p_current = st.session_state.get("presets", [None]*5)[1] if "presets" in st.session_state else None
            else:
                p_current = st.session_state.get("presets", [None]*5)[0] if "presets" in st.session_state else None

            if p_current:
                rsi_low_cur = int(p_current.get("rsi_low", rsi_low))
                rsi_high_cur = int(p_current.get("rsi_high", rsi_high))
                atr_cur = float(p_current.get("atr_threshold", atr_threshold))
                hi_cur = (int(p_current.get("hi_short", hi_pair[0])), int(p_current.get("hi_long", hi_pair[1])))
                lo_cur = (int(p_current.get("lo_short", lo_pair[0])), int(p_current.get("lo_long", lo_pair[1])))
                st.info(f"{timeframe}策略使用通用方案: {p_current.get('name', '未命名')}")
            else:
                rsi_low_cur, rsi_high_cur, atr_cur, hi_cur, lo_cur = rsi_low, rsi_high, atr_threshold, hi_pair, lo_pair

        needed_ma = list(set([hi_cur[0], hi_cur[1], lo_cur[0], lo_cur[1]]))
        df = calculate_indicators(df, ma_lengths=needed_ma)
        signal, reason, indicators = generate_signal(df, rsi_low=rsi_low_cur, rsi_high=rsi_high_cur, atr_threshold=atr_cur, hi_pair=hi_cur, lo_pair=lo_cur)
        
        # 将最新信号存入 session_state 供分析模块使用
        st.session_state["latest_signal_data"] = {
            "signal": signal,
            "reason": reason,
            "rsi": indicators.get("rsi", 0) if indicators else 0,
            "atr": indicators.get("atr", 0) if indicators else 0
        }
        
        st.divider()
        
        # 买卖信号
        col1, col2 = st.columns([1, 2])
        with col1:
            if signal == "买入":
                st.success(f"🟢 {signal}")
            elif signal == "卖出":
                st.error(f"🔴 {signal}")
            else:
                st.warning(f"🟡 {signal}")
            st.info(f"原因: {reason}")
            st.caption(f"当前周期: {st.session_state.get('timeframe', '日线')}")
            alt_sig, alt_reason = None, None
            try:
                if st.session_state.get('timeframe', '日线') == "30分钟":
                    # 上级周期采用“方案一”（槽位1）参数；若无则回退当前参数
                    p_list = st.session_state.get("presets", [None]*5)
                    p_alt = p_list[0] if p_list and len(p_list) > 0 and p_list[0] else {
                        "rsi_low": rsi_low, "rsi_high": rsi_high, "atr_threshold": atr_threshold,
                        "hi_short": hi_pair[0], "hi_long": hi_pair[1], "lo_short": lo_pair[0], "lo_long": lo_pair[1],
                    }
                    rsi_low_alt = int(p_alt.get("rsi_low", rsi_low))
                    rsi_high_alt = int(p_alt.get("rsi_high", rsi_high))
                    atr_alt = float(p_alt.get("atr_threshold", atr_threshold))
                    hi_pair_alt = (int(p_alt.get("hi_short", hi_pair[0])), int(p_alt.get("hi_long", hi_pair[1])))
                    lo_pair_alt = (int(p_alt.get("lo_short", lo_pair[0])), int(p_alt.get("lo_long", lo_pair[1])))
                    df_day_alt = get_history_data_daily(ETF_CODE, 800)
                    if df_day_alt is not None:
                        ma_need = [hi_pair_alt[0], hi_pair_alt[1], lo_pair_alt[0], lo_pair_alt[1]]
                        df_day_alt = calculate_indicators(df_day_alt, ma_lengths=ma_need)
                        s2, r2, _ = generate_signal(
                            df_day_alt,
                            rsi_low=rsi_low_alt, rsi_high=rsi_high_alt,
                            atr_threshold=atr_alt, hi_pair=hi_pair_alt, lo_pair=lo_pair_alt
                        )
                        alt_sig, alt_reason = s2, r2
                        name_suffix = f"（使用方案一）" if st.session_state.get("presets") and st.session_state["presets"][0] else "（使用当前参数）"
                        st.write(f"上级周期(日线)信号: {alt_sig or 'N/A'} {name_suffix}")
                        if alt_reason:
                            st.caption(f"日线原因: {alt_reason}")
                else:
                    # 下级周期采用“方案二”（槽位2）参数；若无则回退当前参数
                    p_list = st.session_state.get("presets", [None]*5)
                    p_alt = p_list[1] if p_list and len(p_list) > 1 and p_list[1] else {
                        "rsi_low": rsi_low, "rsi_high": rsi_high, "atr_threshold": atr_threshold,
                        "hi_short": hi_pair[0], "hi_long": hi_pair[1], "lo_short": lo_pair[0], "lo_long": lo_pair[1],
                    }
                    rsi_low_alt = int(p_alt.get("rsi_low", rsi_low))
                    rsi_high_alt = int(p_alt.get("rsi_high", rsi_high))
                    atr_alt = float(p_alt.get("atr_threshold", atr_threshold))
                    hi_pair_alt = (int(p_alt.get("hi_short", hi_pair[0])), int(p_alt.get("hi_long", hi_pair[1])))
                    lo_pair_alt = (int(p_alt.get("lo_short", lo_pair[0])), int(p_alt.get("lo_long", lo_pair[1])))
                    df_30_alt = get_history_data_30min(ETF_CODE)
                    if df_30_alt is not None and len(df_30_alt) > 0:
                        ma_need = [hi_pair_alt[0], hi_pair_alt[1], lo_pair_alt[0], lo_pair_alt[1]]
                        df_30_alt = calculate_indicators(df_30_alt, ma_lengths=ma_need)
                        s2, r2, _ = generate_signal(
                            df_30_alt,
                            rsi_low=rsi_low_alt, rsi_high=rsi_high_alt,
                            atr_threshold=atr_alt, hi_pair=hi_pair_alt, lo_pair=lo_pair_alt
                        )
                        alt_sig, alt_reason = s2, r2
                        name_suffix = f"（使用方案二）" if st.session_state.get("presets") and len(st.session_state["presets"]) > 1 and st.session_state["presets"][1] else "（使用当前参数）"
                        st.write(f"下级周期(30分钟)信号: {alt_sig or 'N/A'} {name_suffix}")
            except Exception as _e:
                pass
            st.caption("说明：“等待”表示当前周期未触发新买入/卖出信号：若已持仓则维持，若空仓则观望。不同周期可能出现不一致，属正常现象。")
        
        # ============================================
        # 交易日操作预告模块
        # ============================================
        st.divider()
        st.subheader("🔮 交易日操作预告 (临近收盘参考)")
        st.caption("提示：基于当前已完成的K线推算，若本周期收盘价达到以下条件，将触发相应信号。")
        
        fc_col1, fc_col2 = st.columns(2)
        
        def render_forecast_box(container, title, df_target, params, is_daily=True):
            with container:
                st.markdown(f"#### {title}")
                if df_target is None or len(df_target) < 60:
                    st.warning("数据不足")
                    return
                
                # 计算指标
                p_atr = float(params.get("atr_threshold", 1.5))
                p_rsi_l = int(params.get("rsi_low", 40))
                p_rsi_h = int(params.get("rsi_high", 70))
                p_hi = (int(params.get("hi_short", 10)), int(params.get("hi_long", 20)))
                p_lo = (int(params.get("lo_short", 20)), int(params.get("lo_long", 60)))
                
                needed = list(set([p_hi[0], p_hi[1], p_lo[0], p_lo[1]]))
                df_calc = calculate_indicators(df_target.copy(), ma_lengths=needed)
                
                res = calculate_forecast(df_calc, hi_pair=p_hi, lo_pair=p_lo, atr_threshold=p_atr, rsi_low=p_rsi_l, rsi_high=p_rsi_h)
                
                if not res:
                    st.warning("无法计算")
                    return
                    
                cur_p = res['current_price']
                if realtime_data:
                    cur_p = float(realtime_data.get("close", cur_p))
                
                # 显示关键价位
                st.write(f"当前价格: **{cur_p:.3f}**")

                # 获取当前实际均线值（用于显示对比）
                short_ma_name = f"MA{res['n_short']}"
                long_ma_name = f"MA{res['n_long']}"
                current_short_ma = df_calc[short_ma_name].iloc[-1]
                current_long_ma = df_calc[long_ma_name].iloc[-1]

                # 买入条件分析
                # 1. Price > ShortMA
                # 2. ShortMA > LongMA
                # 3. RSI in [L, H]
                # 4. ROC20 > 0 (ROC20 > 0 <=> Price > Close[i-20])

                # ROC20 Check
                base_roc_price = df_calc['Close'].iloc[-21]

                # 构建条件列表
                cond_buy = []
                cond_buy.append(f"价格 > {res['p_cross_short']:.3f} (站上{res['n_short']}均线，当前{short_ma_name}={current_short_ma:.3f})")
                cond_buy.append(f"价格 > {res['p_ma_gold']:.3f} (均线金叉，当前{short_ma_name}={current_short_ma:.3f} {'>' if current_short_ma > current_long_ma else '<'} {long_ma_name}={current_long_ma:.3f})" if cur_p <= res['p_ma_gold'] else f"价格 > {res['p_ma_gold']:.3f} (维持金叉，当前{short_ma_name}={current_short_ma:.3f} {'>' if current_short_ma > current_long_ma else '<'} {long_ma_name}={current_long_ma:.3f})")
                cond_buy.append(f"价格 ∈ [{res['p_rsi_low']:.3f}, {res['p_rsi_high']:.3f}] (RSI达标，当前RSI={df_calc['RSI'].iloc[-1]:.2f})")
                cond_buy.append(f"价格 > {base_roc_price:.3f} (ROC20转正，当前ROC20={df_calc['ROC20'].iloc[-1]:.4f})")
                
                with st.expander("🟢 触发买入条件 (需同时满足)", expanded=True):
                    for c in cond_buy:
                        st.write(f"- {c}")
                    # 综合判定
                    # 粗略计算：需要价格 > max(cross_short, ma_gold, rsi_low_price, roc_base) AND 价格 < rsi_high_price
                    req_min = max(res['p_cross_short'], res['p_ma_gold'], res['p_rsi_low'], base_roc_price)
                    req_max = res['p_rsi_high']
                    if req_min < req_max:
                        st.markdown(f"**👉 目标区间: ({req_min:.3f}, {req_max:.3f})**")
                        if req_min < cur_p < req_max:
                            st.success("✅ 当前价格满足买入条件")
                        else:
                            diff = req_min - cur_p
                            if diff > 0:
                                st.info(f"需上涨 {diff:.3f} ({diff/cur_p:.2%})")
                            else:
                                st.info(f"价格过高，需回调至 {req_max:.3f} 以下")
                    else:
                        conflict_reason = []
                        if res['p_cross_short'] > req_max: conflict_reason.append(f"站上均线需 {res['p_cross_short']:.2f}")
                        if res['p_ma_gold'] > req_max: conflict_reason.append(f"均线金叉需 {res['p_ma_gold']:.2f}")
                        if res['p_rsi_low'] > req_max: conflict_reason.append(f"RSI达标需 {res['p_rsi_low']:.2f}")
                        if base_roc_price > req_max: conflict_reason.append(f"ROC转正需 {base_roc_price:.2f}")
                        
                        st.error(f"❌ 形态无解：触发买入所需价格将导致 RSI 超买 (> {req_max:.2f})")
                        st.caption(f"冲突详情: {'; '.join(conflict_reason)}")

                # 卖出条件分析
                # 1. Price < ShortMA
                # 2. ShortMA < LongMA
                # 3. RSI > Upper or RSI < Lower
                
                cond_sell = []
                cond_sell.append(f"价格 < {res['p_cross_short']:.3f} (跌破{res['n_short']}均线，当前MA{res['n_short']}={df_calc[f'MA{res["n_short"]}'].iloc[-1]:.3f})")
                cond_sell.append(f"价格 < {res['p_ma_gold']:.3f} (均线死叉，当前MA{res['n_short']}={df_calc[f'MA{res["n_short"]}'].iloc[-1]:.3f} {'>' if df_calc[f'MA{res["n_short"]}'].iloc[-1] > df_calc[f'MA{res["n_long"]}'].iloc[-1] else '<'} MA{res['n_long']}={df_calc[f'MA{res["n_long"]}'].iloc[-1]:.3f})")
                cond_sell.append(f"价格 > {res['p_rsi_upper']:.3f} (RSI超买，当前RSI={df_calc['RSI'].iloc[-1]:.2f})")
                cond_sell.append(f"价格 < {res['p_rsi_lower']:.3f} (RSI超卖，当前RSI={df_calc['RSI'].iloc[-1]:.2f})")
                
                with st.expander("🔴 触发卖出条件 (任一满足)", expanded=True):
                    for c in cond_sell:
                        st.write(f"- {c}")
                    
                    # 综合判定
                    # 危险区域：< max_bearish_trigger OR > bullish_exhaustion
                    # Sell if P < p_cross_short OR P < p_ma_gold OR P < p_rsi_lower OR P > p_rsi_upper
                    # Effectively: P < max(p_cross_short, p_ma_gold, p_rsi_lower) OR P > p_rsi_upper
                    
                    sell_trigger_low = max(res['p_cross_short'], res['p_ma_gold'], res['p_rsi_lower'])
                    sell_trigger_high = res['p_rsi_upper']
                    
                    st.markdown(f"**👉 警戒线: < {sell_trigger_low:.3f} 或 > {sell_trigger_high:.3f}**")
                    
                    if cur_p < sell_trigger_low:
                        st.error(f"⚠️ 当前价格触发卖出 (跌破 {sell_trigger_low:.3f})")
                    elif cur_p > sell_trigger_high:
                        st.error(f"⚠️ 当前价格触发卖出 (RSI超买 > {sell_trigger_high:.3f})")
                    else:
                        safe_margin_down = cur_p - sell_trigger_low
                        safe_margin_up = sell_trigger_high - cur_p
                        st.success(f"安全持有中 (下方空间 {safe_margin_down:.3f}, 上方空间 {safe_margin_up:.3f})")

        # 获取参数 - 优先使用专属策略
        stock_strategy = load_stock_strategy(ETF_CODE)

        # 1. 日线参数 - 优先使用专属策略
        if stock_strategy and "daily" in stock_strategy:
            p_day = stock_strategy["daily"]
        elif "presets" in st.session_state and len(st.session_state["presets"]) > 0 and st.session_state["presets"][0]:
            p_day = st.session_state["presets"][0]
        else:
            p_day = {"atr_threshold": 1.5, "rsi_low": 40, "rsi_high": 70, "hi_short": 10, "hi_long": 20, "lo_short": 20, "lo_long": 60}

        # 2. 30分钟参数 - 优先使用专属策略
        if stock_strategy and "30min" in stock_strategy:
            p_30m = stock_strategy["30min"]
        elif "presets" in st.session_state and len(st.session_state["presets"]) > 1 and st.session_state["presets"][1]:
            p_30m = st.session_state["presets"][1]
        else:
            p_30m = {"atr_threshold": 1.5, "rsi_low": 40, "rsi_high": 70, "hi_short": 10, "hi_long": 20, "lo_short": 20, "lo_long": 60}

        # 渲染
        # 获取日线数据
        try:
            df_d = get_history_data_daily(ETF_CODE)
            render_forecast_box(fc_col1, "日线策略", df_d, p_day)
        except Exception as e:
            fc_col1.error(f"日线数据加载失败: {e}")
            
        # 获取30分钟数据
        try:
            df_30 = get_history_data_30min(ETF_CODE)
            render_forecast_box(fc_col2, "30分钟策略", df_30, p_30m, is_daily=False)
        except Exception as e:
            fc_col2.error(f"30分钟数据加载失败: {e}")
            
        with col2:
            c1, c2, c3, c4 = st.columns(4)
            if indicators is None:
                st.warning("可用数据不足以生成完整信号，请扩大时间范围或切换周期")
                c1.metric("收盘价", "N/A")
                c2.metric("均线(短)", "N/A")
                c3.metric("均线(长)", "N/A")
                c4.metric("ATR%", "N/A")
            else:
                c1.metric("收盘价", f"{indicators['price']:.3f}")
                c2.metric(f"均线({indicators['short_ma']})", f"{indicators['short_ma_val']:.3f}")
                c3.metric(f"均线({indicators['long_ma']})", f"{indicators['long_ma_val']:.3f}")
                c4.metric("ATR%", f"{indicators['atr_pct']:.2f}%")
        
        c5, c6 = st.columns(2)
        if indicators is None:
            c5.metric("RSI(14)", "N/A")
            c6.metric("ROC20", "N/A")
        else:
            c5.metric("RSI(14)", f"{indicators['rsi']:.1f}")
            c6.metric("ROC20", f"{indicators['roc20']*100:.2f}%")
        
        # 价格走势图
        st.divider()
        st.caption(f"可用K线数量：{len(df)}，列：{', '.join(list(df.columns))}")
        view_days = st.slider("显示范围(天)", 60, 1500, 250)
        col_pad, col_trim, col_pct = st.columns([1, 1, 2])
        pad_pct = col_pad.slider("纵轴边距(%)", 0, 20, 8, 1)
        use_trim = col_trim.checkbox("剔除异常极值", value=True)
        trim_pct = col_pct.slider("极值剔除百分位(%)", 0, 10, 2, 1)
        cutoff_view = df["Date"].max() - pd.Timedelta(days=view_days)
        df_view = df[df["Date"] >= cutoff_view].copy()

        # 过滤非交易日（周末和法定节假日）
        df_view['Date'] = pd.to_datetime(df_view['Date'])
        df_view = df_view[df_view['Date'].apply(is_trading_day)].copy()

        # 过滤非交易时段（仅30分钟数据）
        if timeframe == "30分钟":
            df_view['hour'] = df_view['Date'].dt.hour
            df_view['minute'] = df_view['Date'].dt.minute
            # A股交易时间: 9:30-11:30, 13:00-15:00
            morning_session = (df_view['hour'] == 9) & (df_view['minute'] >= 30) | (df_view['hour'] == 10) | (df_view['hour'] == 11) & (df_view['minute'] <= 30)
            afternoon_session = (df_view['hour'] == 13) | (df_view['hour'] == 14) | (df_view['hour'] == 15) & (df_view['minute'] == 0)
            df_view = df_view[morning_session | afternoon_session].copy()
            df_view = df_view.drop(columns=['hour', 'minute'])
        fig = go.Figure()
        
        # K线图 - 添加hover显示日期和价格
        fig.add_trace(go.Candlestick(
            x=df_view['Date'],
            open=df_view['Open'],
            high=df_view['High'],
            low=df_view['Low'],
            close=df_view['Close'],
            name='K线',
            hovertext=[
                f"日期: {row['Date'].strftime('%Y-%m-%d %H:%M') if isinstance(row['Date'], pd.Timestamp) else str(row['Date'])}<br>"
                f"开盘: {row['Open']:.3f}<br>"
                f"最高: {row['High']:.3f}<br>"
                f"最低: {row['Low']:.3f}<br>"
                f"收盘: {row['Close']:.3f}<br>"
                f"成交量: {row.get('Volume', 0):,.0f}<br>"
                f"RSI(14): {row.get('RSI', 0):.2f}<br>"
                f"ATR%: {row.get('ATR_Pct', 0):.2f}%"
                for _, row in df_view.iterrows()
            ],
            hoverinfo='text'
        ))
        
        # 均线（根据参数绘制两种市场情形用到的均线）
        ma_set = []
        for n in sorted(set([hi_pair[0], hi_pair[1], lo_pair[0], lo_pair[1]])):
            ma_set.append(n)
        color_cycle = {hi_pair[0]: 'orange', hi_pair[1]: 'gold', lo_pair[0]: 'blue', lo_pair[1]: 'purple'}
        for n in ma_set:
            col = f"MA{n}"
            color = color_cycle.get(n, 'gray')
            fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view[col], name=col, line=dict(color=color, width=1)))

        show_pts = st.checkbox("显示买卖点", value=True)
        if show_pts:
            lookback_days = st.slider("标记范围(天)", 30, 600, 200)
            # 为确保标记与当前K线完全对齐，基于当前视窗数据计算买卖点
            bx, by, sx, sy = compute_signal_points(df_view, rsi_low=rsi_low, rsi_high=rsi_high, atr_threshold=atr_threshold, hi_pair=hi_pair, lo_pair=lo_pair)
            cutoff = df_view["Date"].max() - pd.Timedelta(days=lookback_days)
            left = df_view["Date"].min()
            right = df_view["Date"].max()
            bx_f = [x for x in bx if x >= cutoff]
            by_f = [y for x, y in zip(bx, by) if x >= cutoff]
            sx_f = [x for x in sx if x >= cutoff]
            sy_f = [y for x, y in zip(sx, sy) if x >= cutoff]
            bx_f = [x for x in bx_f if left <= x <= right]
            by_f = [y for x, y in zip(bx_f, by_f) if left <= x <= right]
            sx_f = [x for x in sx_f if left <= x <= right]
            sy_f = [y for x, y in zip(sx_f, sy_f) if left <= x <= right]
            if bx_f:
                fig.add_trace(go.Scatter(x=bx_f, y=by_f, mode="markers", name="买点", marker=dict(symbol="triangle-up", color="lime", size=10, line=dict(width=0))))
            if sx_f:
                fig.add_trace(go.Scatter(x=sx_f, y=sy_f, mode="markers", name="卖点", marker=dict(symbol="triangle-down", color="red", size=10, line=dict(width=0))))
        
        if use_trim and len(df_view) > 5 and trim_pct > 0:
            y_min = float(np.nanpercentile(df_view["Low"], trim_pct))
            y_max = float(np.nanpercentile(df_view["High"], 100 - trim_pct))
        else:
            y_min = float(df_view["Low"].min())
            y_max = float(df_view["High"].max())
        span = y_max - y_min if y_max > y_min else 0
        pad = span * (pad_pct / 100.0) if span > 0 else max(y_min * 0.02, 0.01)
        fig.update_layout(
            title='159915 价格走势图',
            xaxis_title='日期',
            yaxis_title='价格',
            height=500,
            template='plotly_dark',
            xaxis_range=[df_view["Date"].min(), df_view["Date"].max()],
            yaxis_range=[max(y_min - pad, 0), y_max + pad],
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(0,0,0,0.8)',
                font=dict(color='white', size=12),
                bordercolor='white'
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
        
        st.divider()
        with st.expander("📘 指标说明", expanded=False):
            st.markdown(
                "- MA10/MA20/MA60：收盘价的简单移动均线，用于刻画短/中/长趋势\n"
                "- ATR%：真实波动幅度均值占比。TR=max(High-Low, |High-PrevClose|, |Low-PrevClose|)，ATR为14期TR均值，ATR%=ATR/Close×100\n"
                "- RSI(14)：相对强弱指数。RS=14期平均上涨幅/14期平均下跌幅，RSI=100-100/(1+RS)。本实现按14期均值计算\n"
                "- ROC5/ROC20：动量指标，ROCn=(Close/Close(n日前)-1)\n"
                "- 交易规则摘要：当价格>短均线>长均线且RSI∈[侧边栏上下限]且ROC20>0时买入；价格跌破短均线或均线死叉，或RSI高于上限+10/低于下限-10时卖出；短/长均线由ATR%与阈值决定"
            )
        with st.expander("📜 交易规则（详细）", expanded=False):
            # 获取方案参数
            p_d = None
            if "presets" in st.session_state and len(st.session_state["presets"]) > 0 and st.session_state["presets"][0]:
                p_d = st.session_state["presets"][0]
            p_30 = None
            if "presets" in st.session_state and len(st.session_state["presets"]) > 1 and st.session_state["presets"][1]:
                p_30 = st.session_state["presets"][1]
                
            def fmt_p(p, default):
                if not p: return default
                return f"ATR>{p.get('atr_threshold', 1.5)}%用MA{p.get('hi_short', 10)}/{p.get('hi_long', 20)}，否则用MA{p.get('lo_short', 20)}/{p.get('lo_long', 60)}；RSI[{p.get('rsi_low', 40)},{p.get('rsi_high', 70)}]"

            st.markdown(f"""
            **日线策略 (基于方案一)**
            - **均线系统**: {fmt_p(p_d, "ATR>1.5%用MA10/20，否则用MA20/60")}
            - **RSI区间**: [{p_d.get('rsi_low', 40) if p_d else 40}, {p_d.get('rsi_high', 70) if p_d else 70}]
            - **买入条件**: 收盘价>短均线>长均线，且RSI在区间内，且ROC20>0
            - **卖出条件**: 收盘价<短均线，或均线死叉，或RSI超买(>{int(p_d.get('rsi_high', 70) if p_d else 70)+10})/超卖(<{int(p_d.get('rsi_low', 40) if p_d else 40)-10})
            
            **30分钟策略 (基于方案二)**
            - **均线系统**: {fmt_p(p_30, "ATR>1.5%用MA10/20，否则用MA20/60")}
            - **RSI区间**: [{p_30.get('rsi_low', 40) if p_30 else 40}, {p_30.get('rsi_high', 70) if p_30 else 70}]
            - **买入条件**: 同上，基于30分钟K线
            - **卖出条件**: 同上，基于30分钟K线
            
            **通用规则**
            - **持仓**: 全仓进出，不加仓不分批
            - **回测**: 含180天预热期，ETF手续费已扣除（佣金万1.2双向，免印花税）
            """)
        fig_rsi = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rsi.add_trace(go.Candlestick(
            x=df_view['Date'], open=df_view['Open'], high=df_view['High'],
            low=df_view['Low'], close=df_view['Close'], name='K线(叠加)', opacity=0.25
        ), secondary_y=True)
        fig_rsi.add_trace(go.Scatter(x=df_view["Date"], y=df_view["RSI"], name="RSI(14)", line=dict(color="cyan", width=1.5)), secondary_y=False)
        fig_rsi.add_trace(go.Scatter(x=df_view["Date"], y=[70]*len(df_view), name="RSI=70", line=dict(color="red", width=1, dash="dash")), secondary_y=False)
        fig_rsi.add_trace(go.Scatter(x=df_view["Date"], y=[30]*len(df_view), name="RSI=30", line=dict(color="lime", width=1, dash="dash")), secondary_y=False)
        fig_rsi.update_layout(title="RSI 指标（叠加K线）", height=260, template="plotly_dark")
        fig_rsi.update_yaxes(title_text="RSI", secondary_y=False)
        fig_rsi.update_yaxes(showticklabels=False, secondary_y=True)
        st.plotly_chart(fig_rsi, use_container_width=True, config={"scrollZoom": True})
        st.caption("RSI：相对强弱指数，RSI=100-100/(1+RS)，RS=平均上涨幅/平均下跌幅。高位可能过热，低位可能超卖。")
        
        fig_atr = make_subplots(specs=[[{"secondary_y": True}]])
        fig_atr.add_trace(go.Candlestick(
            x=df_view['Date'], open=df_view['Open'], high=df_view['High'],
            low=df_view['Low'], close=df_view['Close'], name='K线(叠加)', opacity=0.25
        ), secondary_y=True)
        fig_atr.add_trace(go.Scatter(x=df_view["Date"], y=df_view["ATR_Pct"], name="ATR%", line=dict(color="magenta", width=1.5)), secondary_y=False)
        fig_atr.update_layout(title="ATR 波动率(%)（叠加K线）", height=240, template="plotly_dark")
        fig_atr.update_yaxes(title_text="ATR%", secondary_y=False)
        fig_atr.update_yaxes(showticklabels=False, secondary_y=True)
        st.plotly_chart(fig_atr, use_container_width=True, config={"scrollZoom": True})
        st.caption("ATR%：真实波动幅百分比。TR=max(H-L, |H-昨收|, |L-昨收|)，ATR=TR的14期均值，ATR%=ATR/收盘×100。")
        
        fig_roc = make_subplots(specs=[[{"secondary_y": True}]])
        fig_roc.add_trace(go.Candlestick(
            x=df_view['Date'], open=df_view['Open'], high=df_view['High'],
            low=df_view['Low'], close=df_view['Close'], name='K线(叠加)', opacity=0.25
        ), secondary_y=True)
        fig_roc.add_trace(go.Bar(x=df_view["Date"], y=df_view["ROC5"]*100, name="ROC5(%)", marker_color="gray"), secondary_y=False)
        fig_roc.add_trace(go.Scatter(x=df_view["Date"], y=df_view["ROC20"]*100, name="ROC20(%)", line=dict(color="gold", width=1.5)), secondary_y=False)
        fig_roc.update_layout(title="ROC 动量(%)（叠加K线）", height=260, template="plotly_dark")
        fig_roc.update_yaxes(title_text="%", secondary_y=False)
        fig_roc.update_yaxes(showticklabels=False, secondary_y=True)
        st.plotly_chart(fig_roc, use_container_width=True, config={"scrollZoom": True})
        st.caption("ROC：变动率，ROC(n)=收盘/收盘(n期前)-1，衡量相对n期前的涨跌动能。")

with tab2:
    if df is not None:
        def periods_per_year():
            tf = st.session_state.get('timeframe', "日线")
            if tf == "30分钟":
                # A股交易约4小时/天 → 8个30分钟bar，年约252个交易日
                return 252 * 8
            return 252
        range_option = st.selectbox(
            "时间区间",
            ["最近一个月", "最近两个月", "最近三个月", "最近半年", "最近一年", "最近两年", "最近三年", "全部", "自定义"],
            index=7
        )
        start_ts = df["Date"].min()
        end_ts = df["Date"].max()
        now_ts = pd.Timestamp(datetime.now().date())
        if range_option != "自定义":
            if range_option == "最近一个月":
                start_ts = now_ts - pd.DateOffset(months=1)
            elif range_option == "最近两个月":
                start_ts = now_ts - pd.DateOffset(months=2)
            elif range_option == "最近三个月":
                start_ts = now_ts - pd.DateOffset(months=3)
            elif range_option == "最近半年":
                start_ts = now_ts - pd.DateOffset(months=6)
            elif range_option == "最近一年":
                start_ts = now_ts - pd.DateOffset(years=1)
            elif range_option == "最近两年":
                start_ts = now_ts - pd.DateOffset(years=2)
            elif range_option == "最近三年":
                start_ts = now_ts - pd.DateOffset(years=3)
            elif range_option == "全部":
                start_ts = df["Date"].min()
                end_ts = df["Date"].max()
            else:
                start_ts = df["Date"].min()
                end_ts = df["Date"].max()
        else:
            col_s, col_e = st.columns(2)
            s_val = col_s.date_input("起始日期", value=df["Date"].min().date())
            e_val = col_e.date_input("结束日期", value=df["Date"].max().date())
            start_ts = pd.to_datetime(s_val)
            end_ts = pd.to_datetime(e_val)
        st.session_state["range_start"] = start_ts
        st.session_state["range_end"] = end_ts
        warmup_days = 180
        df_calc_range = df[(df["Date"] >= (start_ts - pd.Timedelta(days=warmup_days))) & (df["Date"] <= end_ts)].copy()
        if len(df_calc_range) < 60:
            df_calc_range = df.copy()
        df_bt, trades, cumret_bh, cumret_strat = backtest_strategy(df_calc_range.copy(), rsi_low=rsi_low, rsi_high=rsi_high, atr_threshold=atr_threshold, hi_pair=hi_pair, lo_pair=lo_pair)
        st.session_state["trades_bt"] = trades
        st.session_state["bt_params"] = {"rsi_low": rsi_low, "rsi_high": rsi_high, "atr": atr_threshold}
        mask_range = (df_bt["Date"] >= start_ts) & (df_bt["Date"] <= end_ts)
        sub_bh_all = cumret_bh[mask_range]
        sub_strat_all = cumret_strat[mask_range]
        if len(sub_bh_all) > 0:
            base_bh = float(sub_bh_all.iloc[0]) if not pd.isna(sub_bh_all.iloc[0]) else 1.0
            base_strat = float(sub_strat_all.iloc[0]) if not pd.isna(sub_strat_all.iloc[0]) else 1.0
        else:
            base_bh = 1.0
            base_strat = 1.0
        plot_dates = df_bt.loc[mask_range, "Date"]
        plot_bh = (sub_bh_all / base_bh - 1) * 100
        plot_strat = (sub_strat_all / base_strat - 1) * 100
        
        # 收益对比图
        fig = go.Figure()
        
        if show_bh:
            fig.add_trace(go.Scatter(
                x=plot_dates, 
                y=plot_bh,
                name='买入持有',
                line=dict(color='gray', width=2),
                hovertemplate="%{y:.2f}%%"
            ))
        
        if show_strategy:
            fig.add_trace(go.Scatter(
                x=plot_dates, 
                y=plot_strat,
                name='策略收益',
                line=dict(color='green', width=2),
                hovertemplate="%{y:.2f}%%"
            ))
        
        fig.update_layout(
            title=f'收益对比曲线 ({timeframe})',
            xaxis_title='日期',
            yaxis_title='累计收益(%)',
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            hoverdistance=50
        )
        fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1)
        fig.update_yaxes(showspikes=True, spikethickness=1)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
        
        # 回测统计
        col1, col2, col3, col4 = st.columns(4)
        bh_return = float(plot_bh.iloc[-1]) if len(plot_bh) else 0.0
        strat_return = float(plot_strat.iloc[-1]) if len(plot_strat) else 0.0
        
        col1.metric("买入持有收益", f"{bh_return:.2f}%")
        col2.metric("策略收益", f"{strat_return:.2f}%")
        
        # 计算夏普比率
        sub_returns_bh = df_bt.loc[mask_range, 'BuyHoldReturn'].fillna(0)
        sub_returns_strat = df_bt.loc[mask_range, 'StrategyReturn'].fillna(0)
        scale = np.sqrt(periods_per_year())
        sharpe_bh = (sub_returns_bh.mean() / sub_returns_bh.std() * scale) if sub_returns_bh.std() > 0 else 0
        sharpe_strat = (sub_returns_strat.mean() / sub_returns_strat.std() * scale) if sub_returns_strat.std() > 0 else 0
        
        col3.metric("夏普比率(持有)", f"{sharpe_bh:.2f}")
        col4.metric("夏普比率(策略)", f"{sharpe_strat:.2f}")
        
        # 最大回撤
        cum_bh_sub = (plot_bh / 100 + 1)
        cum_strat_sub = (plot_strat / 100 + 1)
        dd_bh = ((cum_bh_sub / cum_bh_sub.cummax()) - 1).min() * 100 if len(cum_bh_sub) else 0
        dd_strat = ((cum_strat_sub / cum_strat_sub.cummax()) - 1).min() * 100 if len(cum_strat_sub) else 0
        
        st.write(f"最大回撤 - 买入持有: **{dd_bh:.2f}%** | 策略: **{dd_strat:.2f}%**")

        # 手续费说明
        st.caption("💡 回测已扣除ETF交易手续费：佣金万1.2双向收取，免印花税和过户费")

        # 交易统计
        st.divider()
        st.subheader("📋 交易记录")
        
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['pnl_pct'] = trades_df['pnl'] * 100
            trades_df = trades_df[(trades_df['exit_date'] >= start_ts) & (trades_df['entry_date'] <= end_ts)]
            display_cols = ['entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct']
            rename_map = {
                'entry_date': '买入日期',
                'entry_price': '买入价',
                'exit_date': '卖出日期',
                'exit_price': '卖出价',
                'pnl_pct': '收益'
            }
            df_disp = trades_df[display_cols].rename(columns=rename_map)
            st.dataframe(
                df_disp,
                use_container_width=True,
                column_config={
                    '买入价': st.column_config.NumberColumn(format="%.3f"),
                    '卖出价': st.column_config.NumberColumn(format="%.3f"),
                    '收益': st.column_config.NumberColumn(format="%.2f%%")
                }
            )
            
            wins = [t for t in trades if (t['pnl'] > 0 and t['exit_date'] >= start_ts and t['entry_date'] <= end_ts)]
            losses = [t for t in trades if (t['pnl'] <= 0 and t['exit_date'] >= start_ts and t['entry_date'] <= end_ts)]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("交易次数", len(trades))
            col2.metric("胜率", f"{len(wins)/len(trades)*100:.1f}%")
            col3.metric("盈亏比", 
                f"{abs(sum([t['pnl'] for t in wins])/sum([t['pnl'] for t in losses])):.2f}" 
                if losses and sum([t['pnl'] for t in losses]) != 0 else "N/A")

with tab3:
    st.subheader("📋 历史交易记录")
    st.write("这里显示所有历史交易信号...")
    
    # 模拟一些交易记录
    if df is not None:
        use_link = False
        if "range_start" in st.session_state and "range_end" in st.session_state:
            use_link = st.checkbox("跟随回测时间区间", value=True, key="trade_follow_bt")
        start_ts = df["Date"].min()
        end_ts = df["Date"].max()
        if not use_link:
            range_option_tr = st.selectbox(
                "时间区间（交易记录）",
                ["最近一个月", "最近两个月", "最近三个月", "最近半年", "最近一年", "最近两年", "最近三年", "全部", "自定义"],
                index=7,
                key="trade_range_select"
            )
        else:
            range_option_tr = None
        now_ts = pd.Timestamp(datetime.now().date())
        if use_link:
            start_ts = st.session_state.get("range_start", start_ts)
            end_ts = st.session_state.get("range_end", end_ts)
        elif range_option_tr != "自定义":
            if range_option_tr == "最近一个月":
                start_ts = now_ts - pd.DateOffset(months=1)
            elif range_option_tr == "最近两个月":
                start_ts = now_ts - pd.DateOffset(months=2)
            elif range_option_tr == "最近三个月":
                start_ts = now_ts - pd.DateOffset(months=3)
            elif range_option_tr == "最近半年":
                start_ts = now_ts - pd.DateOffset(months=6)
            elif range_option_tr == "最近一年":
                start_ts = now_ts - pd.DateOffset(years=1)
            elif range_option_tr == "最近两年":
                start_ts = now_ts - pd.DateOffset(years=2)
            elif range_option_tr == "最近三年":
                start_ts = now_ts - pd.DateOffset(years=3)
            elif range_option_tr == "全部":
                start_ts = df["Date"].min()
                end_ts = df["Date"].max()
        else:
            col_s2, col_e2 = st.columns(2)
            s_val2 = col_s2.date_input("起始日期", value=df["Date"].min().date(), key="trade_start")
            e_val2 = col_e2.date_input("结束日期", value=df["Date"].max().date(), key="trade_end")
            start_ts = pd.to_datetime(s_val2)
            end_ts = pd.to_datetime(e_val2)

        trades_table_df = None
        if use_link and ("trades_bt" in st.session_state):
            tlist = st.session_state["trades_bt"]
            if isinstance(tlist, list) and tlist:
                df_t = pd.DataFrame(tlist)
                df_t = df_t[(df_t['exit_date'] >= start_ts) & (df_t['entry_date'] <= end_ts)]
                df_t = df_t.copy()
                df_t['pnl_pct'] = df_t['pnl'] * 100
                trades_table_df = df_t
        if trades_table_df is None:
            df_calc = calculate_indicators(df.copy())
            trades_list = []
            in_pos = False
            entry_p = 0
            for i in range(60, len(df_calc)):
                row = df_calc.iloc[i]
                short_ma, long_ma = select_ma_by_atr(row, atr_threshold)
                buy = (row['Close'] > row[short_ma]) and (row[short_ma] > row[long_ma]) and (row['RSI'] >= rsi_low) and (row['RSI'] <= rsi_high) and (row['ROC20'] > 0)
                rsi_upper = min(100, rsi_high + 10)
                rsi_lower = max(0, rsi_low - 10)
                sell = (row['Close'] < row[short_ma]) or (row[short_ma] < row[long_ma]) or (row['RSI'] > rsi_upper) or (row['RSI'] < rsi_lower)
                dt = row['Date']
                if not in_pos and buy:
                    in_pos = True
                    entry_p = row['Close']
                    if (dt >= start_ts) and (dt <= end_ts):
                        trades_list.append({
                            '日期': dt.strftime('%Y-%m-%d'),
                            '类型': '买入',
                            '价格': f"{entry_p:.3f}",
                            '信号': f"{short_ma}>{long_ma}, RSI={row['RSI']:.0f}"
                        })
                elif in_pos and sell:
                    in_pos = False
                    exit_p = row['Close']
                    pnl = (exit_p - entry_p) / entry_p * 100
                    if row['RSI'] > rsi_upper:
                        sell_reason = f"RSI超买={row['RSI']:.0f}"
                    elif row['RSI'] < rsi_lower:
                        sell_reason = f"RSI超卖={row['RSI']:.0f}"
                    elif row['Close'] < row[short_ma]:
                        sell_reason = f"价格<{short_ma}"
                    elif row[short_ma] < row[long_ma]:
                        sell_reason = f"{short_ma}<{long_ma}死叉"
                    else:
                        sell_reason = "条件触发卖出"
                    if (dt >= start_ts) and (dt <= end_ts):
                        trades_list.append({
                            '日期': dt.strftime('%Y-%m-%d'),
                            '类型': '卖出',
                            '价格': f"{exit_p:.3f}",
                            '信号': sell_reason,
                            '收益': f"{pnl:+.2f}%"
                        })
            if trades_list:
                trades_table_df = pd.DataFrame(trades_list)
        if trades_table_df is not None and len(trades_table_df):
            st.dataframe(trades_table_df, use_container_width=True)

# 底部
st.divider()
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_version.json"), "r", encoding="utf-8") as _vf:
        _vinfo = pyjson.load(_vf)
        _ver = _vinfo.get("version", "1.0")
except Exception:
    _ver = "1.0"

# 动态确定数据来源描述
# 1. 实时价格：新浪财经
# 2. 历史数据（日线）：网易/新浪
# 3. 历史数据（30分钟）：Baostock/东方财富/新浪
# 根据当前选择的周期和实际使用的API状态来显示

source_desc = "新浪财经 (实时)"
if timeframe == "30分钟":
    if "last_30m_api_status" in st.session_state:
        status = st.session_state["last_30m_api_status"]
        if "Baostock" in status:
            source_desc += " / 证券宝 (30分钟)"
        elif "Akshare" in status:
            source_desc += " / 东方财富 (30分钟)"
    else:
        source_desc += " / 证券宝/东方财富 (30分钟)"
else:
    source_desc += " / 网易财经 (日线)"

st.caption(f"📊 数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 版本: v{_ver} | 版权所有 © 邱俊斌 | 数据来源: {source_desc} | 回测仅供参考，不构成投资建议")
