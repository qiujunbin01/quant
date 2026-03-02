
import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from itertools import product

# 尝试导入依赖模块
try:
    import stock_prediction_module as spm
except ImportError:
    spm = None

try:
    import stock_analysis_module as sam
except ImportError:
    sam = None

# ============================================
# 复用 etf_monitor.py 的核心策略逻辑 (解耦版本)
# ============================================

# 策略存储管理 (复用 etf_monitor.py 逻辑)
STRATEGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_strategies.json")

def load_stock_strategy(code):
    if not os.path.exists(STRATEGY_FILE):
        return None
    try:
        with open(STRATEGY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(code)
    except:
        return None

def save_stock_strategy(code, strategy_data):
    """保存股票专属策略"""
    all_data = {}
    if os.path.exists(STRATEGY_FILE):
        try:
            with open(STRATEGY_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        except: pass
    
    # Update or add
    if code not in all_data:
        all_data[code] = {}
    
    # 添加时间戳
    strategy_data["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_data[code]["daily"] = strategy_data
    all_data[code]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(STRATEGY_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving strategy for {code}: {e}")

def optimize_selection_strategy(df):
    """
    为该股票生成专属的最优参数组合 (Grid Search)
    针对当前选股策略逻辑：ATR分层均线 + RSI区间 + ROC趋势
    """
    if df is None or len(df) < 120:
        return None
        
    # 1. 预计算所有可能的指标
    # 使用副本以免影响原df
    df_opt = df.copy()
    
    # 确保列名正确 (analyze_stock_strategy 中传入的是小写列名，但在 calculate_indicators 前)
    # 这里我们自己处理一下
    rename_map = {'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}
    df_opt.rename(columns=rename_map, inplace=True)
    
    # MAs: 5, 10, 20, 30, 60
    ma_periods = [5, 10, 20, 30, 60]
    for p in ma_periods:
        df_opt[f'MA{p}'] = df_opt['Close'].rolling(p).mean()
        
    # RSI (14)
    delta = df_opt['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_opt['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR (14) & ATR_Pct
    high = df_opt['High']
    low = df_opt['Low']
    close = df_opt['Close']
    tr = pd.concat([
        high - low, 
        (high - close.shift(1)).abs(), 
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df_opt['ATR'] = tr.rolling(14).mean()
    df_opt['ATR_Pct'] = (df_opt['ATR'] / close) * 100
    
    # ROC20
    df_opt['ROC20'] = df_opt['Close'].pct_change(20)
    
    # Drop NaN
    df_opt = df_opt.dropna()
    if len(df_opt) < 60: return None
    
    # Numpy arrays for speed
    close_arr = df_opt['Close'].values
    rsi_arr = df_opt['RSI'].values
    atr_pct_arr = df_opt['ATR_Pct'].values
    roc_arr = df_opt['ROC20'].values
    
    ma_data = {p: df_opt[f'MA{p}'].values for p in ma_periods}
    
    # Parameter Grid
    # 缩小搜索空间以提高速度
    grid_rsi_low = [30, 35, 40]
    grid_rsi_high = [70, 75, 80]
    grid_atr = [1.5, 2.0]
    grid_hi_pair = [(5, 10), (10, 20)]
    grid_lo_pair = [(20, 60), (30, 60)]
    
    best_score = -9999
    best_params = None
    
    # 生成组合
    combinations = list(product(grid_rsi_low, grid_rsi_high, grid_atr, grid_hi_pair, grid_lo_pair))
    
    # 遍历搜索
    for r_low, r_high, atr_th, hi_p, lo_p in combinations:
        hi_s, hi_l = hi_p
        lo_s, lo_l = lo_p
        
        # 1. 确定每行使用的 MA (Vectorized)
        # mask: True if High Volatility
        mask_hi = atr_pct_arr > atr_th
        
        # Select short MA
        short_ma = np.where(mask_hi, ma_data[hi_s], ma_data[lo_s])
        # Select long MA
        long_ma = np.where(mask_hi, ma_data[hi_l], ma_data[lo_l])
        
        # 2. 生成信号
        # 买入条件: Price > Short > Long AND RSI in [low, high] AND ROC > 0
        buy_sig = (close_arr > short_ma) & (short_ma > long_ma) & \
                  (rsi_arr >= r_low) & (rsi_arr <= r_high) & (roc_arr > 0)
                  
        # 卖出条件: Price < Short OR Short < Long OR RSI > high+10 OR RSI < low-10
        # 注意: 原始代码 sell_filter = (rsi > rsi_upper) or (rsi < rsi_lower)
        # rsi_upper = r_high + 10, rsi_lower = r_low - 10
        rsi_upper = min(100, r_high + 10)
        rsi_lower = max(0, r_low - 10)
        
        sell_sig = (close_arr < short_ma) | (short_ma < long_ma) | \
                   (rsi_arr > rsi_upper) | (rsi_arr < rsi_lower)
        
        # 3. 快速回测 (Event Driven Loop)
        # 我们只关心买入后的收益表现
        
        buy_indices = np.flatnonzero(buy_sig)
        if len(buy_indices) == 0: continue
        
        sell_indices = np.flatnonzero(sell_sig)
        
        # 合并事件流 (index, type: 1=Buy, -1=Sell)
        events = []
        for i in buy_indices: events.append((i, 1))
        for i in sell_indices: events.append((i, -1))
        events.sort(key=lambda x: x[0])
        
        pos = 0
        entry_price = 0.0
        tx_profits = []
        
        for idx, action in events:
            if pos == 0 and action == 1:
                pos = 1
                entry_price = close_arr[idx]
            elif pos == 1 and action == -1:
                pos = 0
                ret = (close_arr[idx] - entry_price) / entry_price
                tx_profits.append(ret)
                
        # 4. 评分
        if not tx_profits:
            score = 0
        else:
            total_ret = sum(tx_profits)
            win_count = len([x for x in tx_profits if x > 0])
            count = len(tx_profits)
            win_rate = win_count / count
            
            # 评分公式: 侧重总收益，兼顾胜率
            score = total_ret * 0.7 + win_rate * 0.3
            
        if score > best_score:
            best_score = score
            best_params = {
                "rsi_low": r_low,
                "rsi_high": r_high,
                "atr_threshold": atr_th,
                "hi_short": hi_s,
                "hi_long": hi_l,
                "lo_short": lo_s,
                "lo_long": lo_l,
                "score": score,
                "win_rate": win_rate if 'win_rate' in locals() else 0
            }
            
    return best_params

def calculate_indicators(df, ma_lengths=None):
    """计算技术指标 (MA, ATR, RSI, ROC)"""
    default_ma = [10, 20, 60]
    ma_lengths = sorted(set((ma_lengths or []) + default_ma))
    
    # 确保有日期列并排序
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    # 列名标准化
    if 'close' in df.columns and 'Close' not in df.columns:
        df = df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'})
        
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
    """生成买卖信号"""
    if df is None or len(df) < 60:
        return None, "数据不足", {}
    
    last_row = df.iloc[-1]
    
    short_ma, long_ma = select_ma_by_atr(last_row, atr_threshold, hi_pair=hi_pair, lo_pair=lo_pair)
    short_ma_val = last_row[short_ma]
    long_ma_val = last_row[long_ma]
    
    price = last_row['Close']
    rsi = last_row['RSI']
    roc20 = last_row['ROC20']
    atr_pct = last_row['ATR_Pct']
    
    # 策略核心逻辑 (与主程序保持一致)
    buy_cond = (price > short_ma_val) and (short_ma_val > long_ma_val)
    buy_filter = (rsi >= rsi_low) and (rsi <= rsi_high) and (roc20 > 0)
    
    sell_cond = (price < short_ma_val) or (short_ma_val < long_ma_val)
    rsi_upper = min(100, rsi_high + 10)
    rsi_lower = max(0, rsi_low - 10)
    sell_filter = (rsi > rsi_upper) or (rsi < rsi_lower)
    
    signal = "等待"
    reason = "未触发信号"
    
    if buy_cond and buy_filter:
        signal = "买入"
        reason = f"价格>{short_ma}>{long_ma}，RSI={rsi:.1f}，ROC>0"
    elif sell_cond or sell_filter:
        signal = "卖出"
        if rsi > rsi_upper: reason = f"RSI超买 {rsi:.1f}"
        elif rsi < rsi_lower: reason = f"RSI超卖 {rsi:.1f}"
        else: reason = f"趋势向下 ({short_ma}/{long_ma})"
    
    return signal, reason, {
        "price": price, "rsi": rsi, "atr": atr_pct, "ma_short": short_ma_val, "ma_long": long_ma_val
    }

# ============================================
# AI 选股逻辑
# ============================================

def get_hot_topics_and_stocks_by_ai():
    """
    通过 AI 查找最近一周的热点题材和相关股票
    """
    if sam is None:
        st.error("无法加载分析模块 (stock_analysis_module)，无法调用 AI。")
        return []

    system_prompt = """你是一名资深的A股市场策略分析师。
请分析中国股市最近一周（5个交易日）的市场动态，优先关注以下维度的题材（按重要性排序）：
1. 国家大事/社会热点（如国家级战略、全民热议事件）
2. 政策支持（如近期发布的产业政策、区域规划）
3. 技术进步（如AI算力、半导体、新能源技术突破）
4. 个股利好（如业绩预增、重组、订单落地）

任务要求：
1. 提炼出当前最热门的 3-5 个题材。
2. 为每个题材推荐 3-5 只最相关的龙头股票（必须是A股）。
3. 输出严格的 JSON 格式，不要包含 Markdown 代码块标记（如 ```json），直接返回 JSON 字符串。

JSON 格式示例：
[
  {
    "topic": "题材名称",
    "reason": "推荐理由",
    "stocks": [
      {"code": "600000", "name": "浦发银行"},
      {"code": "000001", "name": "平安银行"}
    ]
  }
]
注意：股票代码必须是 6 位数字。
"""
    
    user_prompt = f"今天是 {datetime.now().strftime('%Y-%m-%d')}，请分析最近一周的A股热点题材和个股。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    with st.spinner("正在调用 AI 分析全网热点题材... (预计 10-20 秒)"):
        response = sam.call_kimi_chat(messages)

    # 检查API返回的是否是错误信息
    if response.startswith("API 调用失败") or response.startswith("API认证失败") or response.startswith("请求超时") or response.startswith("请求发生错误") or response.startswith("JSON解析失败"):
        st.error(f"Kimi API 调用失败: {response}")
        return []

    # 解析 JSON
    try:
        # 清理可能存在的 markdown 标记
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_response)
        # 验证数据结构 - 支持列表或字典格式
        if isinstance(data, dict):
            # 如果返回的是字典，尝试提取topics字段或包装成列表
            if 'topics' in data and isinstance(data['topics'], list):
                data = data['topics']
            elif 'data' in data and isinstance(data['data'], list):
                data = data['data']
            else:
                # 将单个字典包装成列表
                data = [data]

        if not isinstance(data, list):
            st.error(f"AI 返回数据格式错误: 期望列表，得到 {type(data)}")
            return []

        # 验证每个题材的结构
        valid_topics = []
        for topic in data:
            if isinstance(topic, dict) and 'topic' in topic and 'stocks' in topic:
                if isinstance(topic['stocks'], list):
                    valid_topics.append(topic)
        return valid_topics
    except Exception as e:
        st.error(f"AI 返回数据解析失败: {e}")
        with st.expander("查看原始返回内容"):
            st.text(response)
        return []

def analyze_stock_strategy(code, name):
    """
    分析单只股票的日线策略状态
    """
    try:
        # 获取数据 - 优先尝试本地数据
        df = None
        data_source = None
        try:
            import glob
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, "数据")

            # 1. 首先尝试从 股票数据 文件夹中查找CSV文件
            stock_data_dir = os.path.join(data_dir, "股票数据")
            if os.path.exists(stock_data_dir):
                # 转换code格式: 600000 -> sh600000, 000001 -> sz000001
                if code.startswith('6'):
                    stock_prefix = f"sh{code}"
                else:
                    stock_prefix = f"sz{code}"

                csv_files = glob.glob(os.path.join(stock_data_dir, f"*{stock_prefix}*.csv"))
                if csv_files:
                    csv_files.sort(key=os.path.getmtime, reverse=True)
                    # 尝试不同的编码读取CSV
                    try:
                        df = pd.read_csv(csv_files[0], encoding='utf-8')
                    except:
                        try:
                            df = pd.read_csv(csv_files[0], encoding='gbk')
                        except:
                            df = pd.read_csv(csv_files[0], encoding='gb2312')

                    if not df.empty:
                        # 处理30分钟数据 - 转换为日线
                        # 检查第一列是否是时间格式
                        first_col = df.columns[0]
                        if '时间' in first_col or 'date' in first_col.lower():
                            df[first_col] = pd.to_datetime(df[first_col])
                            # 如果是30分钟数据（有具体时间），转换为日线
                            if df[first_col].dt.hour.nunique() > 1:
                                # 提取日期
                                df['date_only'] = df[first_col].dt.date
                                # 按日期聚合为日线数据
                                df_daily = df.groupby('date_only').agg({
                                    df.columns[3]: 'first',  # 开盘价
                                    df.columns[5]: 'max',    # 最高价
                                    df.columns[6]: 'min',    # 最低价
                                    df.columns[4]: 'last',   # 收盘价
                                    df.columns[7]: 'sum',    # 成交量
                                    df.columns[8]: 'sum'     # 成交额
                                }).reset_index()
                                df_daily.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
                                df = df_daily
                            else:
                                # 已经是日线数据，重命名列
                                df = df.rename(columns={
                                    first_col: 'date',
                                    df.columns[3]: 'open',
                                    df.columns[4]: 'close',
                                    df.columns[5]: 'high',
                                    df.columns[6]: 'low',
                                    df.columns[7]: 'volume',
                                    df.columns[8]: 'amount'
                                })
                        data_source = "local_csv"
                        st.info(f"{name}({code}): 使用本地CSV数据 ({len(df)} 条)")

            # 2. 如果没有找到，尝试从根目录查找xlsx文件
            if df is None or df.empty:
                xlsx_files = glob.glob(os.path.join(base_dir, f"*{code}*.xlsx"))
                if xlsx_files:
                    xlsx_files.sort(key=os.path.getmtime, reverse=True)
                    df = pd.read_excel(xlsx_files[0], engine='openpyxl')
                    if not df.empty:
                        data_source = "local_xlsx"
                        st.info(f"{name}({code}): 使用本地Excel数据")
        except Exception as e:
            pass

        # 如果本地没有数据，尝试akshare
        if df is None or df.empty:
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
                if df is not None and not df.empty:
                    data_source = "akshare"
            except Exception as e:
                pass

        if df is None or df.empty or len(df) < 60:
            # 返回一个带提示的失败结果
            return {
                "code": code,
                "name": name,
                "signal": "数据不足",
                "reason": "本地无数据且网络获取失败",
                "score": 0,
                "params": None
            }
            
        # 标准化
        rename_map = {
            '日期': 'date', '开盘': 'open', '收盘': 'close', 
            '最高': 'high', '最低': 'low', '成交量': 'volume', 
            '成交额': 'amount'
        }
        df.rename(columns=rename_map, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        
        # 确保包含 amount 列 (AI预测模型需要)
        if 'amount' not in df.columns:
            if 'close' in df.columns and 'volume' in df.columns:
                df['amount'] = df['close'] * df['volume']
            else:
                df['amount'] = 0.0
        
        # 创建副本用于计算指标 (因为 calculate_indicators 会修改列名为首字母大写)
        df_for_ind = df.copy()
        
        # 尝试加载该股票的专属策略 (日线)
        stock_strategy = load_stock_strategy(code)
        custom_params = None
        
        # 如果没有策略，进行自动优化
        if not stock_strategy:
            # st.write(f"正在为 {name} ({code}) 生成专属策略...") # 可选日志
            opt_res = optimize_selection_strategy(df)
            if opt_res:
                save_stock_strategy(code, opt_res)
                custom_params = opt_res
                # reason append info later
        elif "daily" in stock_strategy:
            custom_params = stock_strategy["daily"]
            
        # 再次检查完整性 (无论是加载的还是新生成的)
        if custom_params:
            required_keys = ["rsi_low", "rsi_high", "atr_threshold", "hi_short", "hi_long", "lo_short", "lo_long"]
            if not all(k in custom_params for k in required_keys):
                custom_params = None
        
        # 准备指标计算所需的均线周期
        needed_ma = []
        if custom_params:
            needed_ma = [
                int(custom_params["hi_short"]), int(custom_params["hi_long"]),
                int(custom_params["lo_short"]), int(custom_params["lo_long"])
            ]
        
        df_for_ind = calculate_indicators(df_for_ind, ma_lengths=needed_ma)
        
        # 生成信号 (使用专属参数或默认参数)
        if custom_params:
            signal, reason, indicators = generate_signal(
                df_for_ind,
                rsi_low=int(custom_params["rsi_low"]),
                rsi_high=int(custom_params["rsi_high"]),
                atr_threshold=float(custom_params["atr_threshold"]),
                hi_pair=(int(custom_params["hi_short"]), int(custom_params["hi_long"])),
                lo_pair=(int(custom_params["lo_short"]), int(custom_params["lo_long"]))
            )
            reason += " (专属策略)"
            # 可选：显示胜率信息
            # if "win_rate" in custom_params:
            #    reason += f" [历史胜率:{custom_params['win_rate']:.0%}]"
        else:
            signal, reason, indicators = generate_signal(df_for_ind)
            reason += " (通用策略)"
        
        return {
            "code": code,
            "name": name,
            "signal": signal,
            "reason": reason,
            "price": indicators['price'],
            "df": df # 返回原始的小写列名数据，供 AI 模型使用
        }
    except Exception as e:
        # print(f"分析 {code} 失败: {e}")
        return None

def verify_stock_prediction(df, code):
    """
    使用 Kronos 模型预测未来一周趋势
    """
    if spm is None:
        return 0, "AI模块不可用"
        
    try:
        predictor = spm.load_model()
        if not predictor:
            return 0, "模型加载失败"
            
        # 准备数据
        lookback = 400
        pred_len = 5 # 预测一周 (5个交易日)
        
        x_df, x_timestamp = spm.prepare_inputs(df, lookback, "daily")
        last_date = df["date"].iloc[-1]
        y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)
        
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=pd.Series(y_timestamp),
            pred_len=pred_len, T=0.9, top_p=0.9, sample_count=1, verbose=False
        )
        
        if pred_df is not None and not pred_df.empty:
            start_p = pred_df.iloc[0]['close']
            end_p = pred_df.iloc[-1]['close']
            change_pct = (end_p - start_p) / start_p * 100
            return change_pct, f"未来5日涨幅: {change_pct:.2f}%"
            
    except Exception as e:
        return 0, f"预测异常: {e}"
        
    return 0, "无结果"

def render_selection_page():
    st.header("🎯 智能题材选股")
    st.info("本模块通过 AI 分析全网资讯，捕捉政策与社会热点，结合技术面策略与 AI 股价预测，为您推荐“三位一体”的潜力牛股。")
    
    # 检查是否已有今日选股结果
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"selection_result_{today_str}"
    
    has_cache = cache_key in st.session_state
    
    # 按钮逻辑：如果有缓存，显示“重新选股”；否则显示“开始选股”
    btn_label = "🔄 重新启动全自动选股" if has_cache else "🚀 启动全自动选股流程"
    
    # 我们使用一个变量来控制是否运行逻辑
    # 1. 点击按钮 -> run = True
    # 2. 有缓存 -> run = False, display = True
    
    # 为了保持 Streamlit 的状态，我们将结果存入 session_state
    
    # 如果点击了按钮，清除旧缓存并重新运行
    # Streamlit 的 button 返回 True 仅在点击的那一帧
    
    # 修正逻辑：
    # 默认显示缓存结果（如果存在）
    # 点击按钮则重新计算并覆盖缓存
    
    # 记录点击状态
    # button 仅在点击时为 True
    
    is_running = False
    
    # 如果点击了按钮
    if st.button(btn_label, type="primary", key="btn_start_selection"):
        is_running = True
        # 清除旧缓存（可选，覆盖即可）
    
    # 如果有缓存且没有点击重新运行，直接显示缓存
    if not is_running and has_cache:
        st.success(f"已加载今日 ({today_str}) 的选股结果")
        display_selection_results(st.session_state[cache_key])
        return

    # 如果需要运行（点击了按钮）
    if is_running:
        # 1. AI 题材挖掘
        hot_topics = get_hot_topics_and_stocks_by_ai()
        
        if not hot_topics:
            return
            
        # ... (中间的扫描逻辑) ...
        # 为了避免代码重复太长，我们将中间逻辑封装一下，或者在这里直接写
        
        st.subheader("🔥 本周热门题材扫描")
        
        # 收集所有候选股票
        all_candidates = []
        for topic in hot_topics:
            # ... (展示题材) ...
            # 这里需要把题材展示也缓存下来吗？最好是。
            # 为了简单，我们将 hot_topics 和 valid_recommendations 都存入缓存对象
            all_candidates.extend(topic['stocks']) # 简化展示逻辑，后面统一存
            
        # 这里为了展示过程，我们还得保留 st.write 等 UI 操作
        # 这是一个流式过程，不仅要计算，还要实时显示进度
        
        # ... (展示题材 UI) ...
        for topic in hot_topics:
            with st.expander(f"📢 {topic['topic']}", expanded=True):
                st.write(topic['reason'])
                cols = st.columns(len(topic['stocks']))
                for idx, stock in enumerate(topic['stocks']):
                    cols[idx].caption(f"{stock['name']} ({stock['code']})")

        st.divider()
        st.subheader("⚡ 深度策略分析与预测")
        
        valid_recommendations = []
        scan_logs = [] # 记录扫描日志
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander("查看详细扫描日志", expanded=True)
        
        # 去重
        unique_candidates = {s['code']: s for s in all_candidates}.values()
        total_scan = len(unique_candidates)
        
        for i, stock in enumerate(unique_candidates):
            code = stock['code']
            name = stock['name']
            
            # 第一步：下载数据
            status_text.text(f"[{i+1}/{total_scan}] {name} ({code}): 正在下载日线数据...")
            
            # 3. 计算日线策略
            res = analyze_stock_strategy(code, name)
            
            if res:
                signal = res['signal']
                reason = res['reason']
                
                # 记录日志
                log_entry = f"**{name} ({code})**: 信号【{signal}】 - {reason}"
                
                # 4. 检查是否为买入信号
                if signal == "买入":
                    # 5. 进行股价预测
                    status_text.text(f"[{i+1}/{total_scan}] {name}: ✅ 技术面符合！正在调用 AI 模型进行股价预测 (耗时操作)...")
                    pred_pct, pred_msg = verify_stock_prediction(res['df'], code)
                    
                    log_entry += f" | 🤖 AI预测: {pred_msg}"
                    
                    if pred_pct > 0: # 预测未来一周上涨
                        res['pred_pct'] = pred_pct
                        res['pred_msg'] = pred_msg
                        # 为了缓存，删除 df 对象
                        if 'df' in res: del res['df']
                        valid_recommendations.append(res)
                        scan_logs.append(f"🟢 {log_entry}")
                    else:
                        scan_logs.append(f"🟡 {log_entry} (AI预测下跌，剔除)")
                else:
                    # 非买入信号，跳过预测
                    status_text.text(f"[{i+1}/{total_scan}] {name}: 信号【{signal}】，跳过AI预测以节省时间。")
                    scan_logs.append(f"⚪ {log_entry}")
                    time.sleep(0.1) #稍微停顿让用户看清状态
            else:
                res = analyze_stock_strategy(code, name)
                if res and res.get('signal') == "数据不足":
                    scan_logs.append(f"🔴 **{name} ({code})**: ⚠️ 本地无数据，网络获取失败")
                else:
                    scan_logs.append(f"🔴 **{name} ({code})**: 数据获取失败或不足")
            
            # 实时更新日志到前端
            if i % 5 == 0 or i == total_scan - 1:
                with log_container:
                    # 倒序显示，最新的在最上面
                    for log in reversed(scan_logs[-5:]):
                        st.markdown(log)
            
            progress_bar.progress((i + 1) / total_scan)
            
        progress_bar.empty()
        status_text.empty()
        
        # 存入缓存
        result_package = {
            "topics": hot_topics,
            "recommendations": valid_recommendations,
            "logs": scan_logs
        }
        st.session_state[cache_key] = result_package
        
        # 显示结果
        display_selection_results(result_package)

def display_selection_results(data):
    """
    展示选股结果 (从缓存或新计算的数据)
    """
    hot_topics = data.get("topics", [])
    valid_recommendations = data.get("recommendations", [])
    scan_logs = data.get("logs", [])
    
    # 这里我们只展示最终推荐列表，题材部分如果已经过去就不重复渲染了，或者放在折叠栏里
    with st.expander("回顾本周热门题材", expanded=False):
        for topic in hot_topics:
            st.markdown(f"**{topic['topic']}**: {topic['reason']}")
            st.caption(", ".join([f"{s['name']}" for s in topic['stocks']]))
            
    with st.expander("查看详细扫描日志", expanded=False):
        for log in scan_logs:
            st.markdown(log)

    if valid_recommendations:
        st.success(f"🎉 筛选完成！共发现 {len(valid_recommendations)} 只符合【题材+策略+预测】三重标准的金股")
        
        # 按预测涨幅排序
        valid_recommendations.sort(key=lambda x: x['pred_pct'], reverse=True)
        
        for stock in valid_recommendations:
            with st.container():
                st.markdown(f"### 🏆 {stock['name']} ({stock['code']})")
                c1, c2, c3 = st.columns(3)
                # price 可能是 float
                st.metric("当前价格", f"{stock['price']:.2f}" if isinstance(stock['price'], (int, float)) else stock['price'])
                c2.metric("日线信号", stock['signal'], help=stock['reason'])
                c3.metric("AI预测(5日)", f"+{stock['pred_pct']:.2f}%", help="Kronos模型预测未来5个交易日涨幅")
                
                if st.button(f"查看 {stock['name']} 详情", key=f"rec_{stock['code']}"):
                    st.session_state["target_code"] = stock['code']
                    st.session_state["pred_symbol"] = stock['code']
                    st.session_state["auto_run_prediction"] = True
                    st.success(f"已切换到 {stock['name']}！")
                st.divider()
    else:
        st.warning("本次扫描未发现同时满足【日线买入信号】且【AI预测上涨】的股票。建议关注题材列表中的股票，等待回调机会。")
