
import streamlit as st
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
from datetime import datetime, timedelta

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

# 延迟导入 torch，避免在模块加载时直接崩溃
torch = None

# Ensure Kronos model can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
kronos_dir = os.path.join(current_dir, "Kronos")
if kronos_dir not in sys.path:
    sys.path.append(kronos_dir)

try:
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError as e:
    # st.error(f"无法导入 Kronos 模型模块。请确保 'Kronos' 文件夹完整且依赖已安装。错误: {e}")
    Kronos = None

# Constants
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"
DEVICE = "cpu"
if torch and torch.cuda.is_available():
    DEVICE = "cuda:0"

@st.cache_resource
def load_model():
    """Load the Kronos model and tokenizer (cached)."""
    if Kronos is None:
        return None
    try:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
        model = Kronos.from_pretrained(MODEL_PRETRAINED)
        predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)
        return predictor
    except Exception as e:
        st.error(f"加载模型出错: {e}")
        return None

def load_stock_data(symbol, period="daily", force_update=False):
    """Fetch stock data from Akshare or local files.

    Args:
        symbol: 股票代码
        period: 数据周期 (daily/30)
        force_update: 是否强制从网络获取最新数据
    """
    max_retries = 3
    df = None

    # 1. 首先尝试从本地文件加载数据（如果不是强制更新）
    if not force_update:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, "数据")
            import glob
            df_local = None

            # 1.1 尝试从 股票数据 文件夹中查找CSV文件
            stock_data_dir = os.path.join(data_dir, "股票数据")
            if os.path.exists(stock_data_dir):
                # 转换code格式
                if symbol.startswith('6'):
                    stock_prefix = f"sh{symbol}"
                else:
                    stock_prefix = f"sz{symbol}"

                csv_files = glob.glob(os.path.join(stock_data_dir, f"*{stock_prefix}*.csv"))
                if csv_files:
                    csv_files.sort(key=os.path.getmtime, reverse=True)
                    # 检查数据是否最新（最后修改日期是否为今天）
                    latest_file = csv_files[0]
                    mtime = datetime.fromtimestamp(os.path.getmtime(latest_file))
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

                    if mtime >= today:
                        st.info(f"使用今日已更新的本地数据: {os.path.basename(latest_file)}")
                    else:
                        st.warning(f"本地数据不是最新的（最后更新: {mtime.strftime('%Y-%m-%d')}），将从网络获取最新数据...")
                        raise Exception("数据需要更新")

                    try:
                        df_local = pd.read_csv(latest_file, encoding='utf-8')
                    except:
                        try:
                            df_local = pd.read_csv(latest_file, encoding='gbk')
                        except:
                            df_local = pd.read_csv(latest_file, encoding='gb2312')

                    if df_local is not None and not df_local.empty:
                        # 处理30分钟数据 - 转换为日线
                        first_col = df_local.columns[0]
                        if '时间' in first_col or 'date' in first_col.lower():
                            df_local[first_col] = pd.to_datetime(df_local[first_col])
                            # 如果是30分钟数据，转换为日线
                            if df_local[first_col].dt.hour.nunique() > 1:
                                df_local['date_only'] = df_local[first_col].dt.date
                                df_daily = df_local.groupby('date_only').agg({
                                    df_local.columns[3]: 'first',  # open
                                    df_local.columns[5]: 'max',    # high
                                    df_local.columns[6]: 'min',    # low
                                    df_local.columns[4]: 'last',   # close
                                    df_local.columns[7]: 'sum',    # volume
                                    df_local.columns[8]: 'sum'     # amount
                                }).reset_index()
                                df_daily.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
                                df_local = df_daily
                            else:
                                df_local = df_local.rename(columns={
                                    first_col: 'date',
                                    df_local.columns[3]: 'open',
                                    df_local.columns[4]: 'close',
                                    df_local.columns[5]: 'high',
                                    df_local.columns[6]: 'low',
                                    df_local.columns[7]: 'volume',
                                    df_local.columns[8]: 'amount'
                                })

            # 1.2 如果没有找到CSV，尝试从根目录查找xlsx文件
            if df_local is None or df_local.empty:
                xlsx_files = glob.glob(os.path.join(base_dir, f"*{symbol}*.xlsx"))
                if xlsx_files:
                    xlsx_files.sort(key=os.path.getmtime, reverse=True)
                    latest_xlsx = xlsx_files[0]
                    mtime = datetime.fromtimestamp(os.path.getmtime(latest_xlsx))
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

                    if mtime >= today:
                        st.info(f"使用今日已更新的本地Excel数据: {os.path.basename(latest_xlsx)}")
                    else:
                        st.warning(f"本地Excel数据不是最新的（最后更新: {mtime.strftime('%Y-%m-%d')}），将从网络获取最新数据...")
                        raise Exception("数据需要更新")
                    df_local = pd.read_excel(latest_xlsx, engine='openpyxl')

                    df_local = pd.read_excel(latest_xlsx, engine='openpyxl')
                    if df_local is not None and not df_local.empty:
                        # 标准化列名
                        col_map = {
                            "Date": "date", "日期": "date", "时间": "date",
                            "Open": "open", "开盘": "open",
                            "Close": "close", "收盘": "close",
                            "High": "high", "最高": "high",
                            "Low": "low", "最低": "low",
                            "Volume": "volume", "成交量": "volume", "总手": "volume",
                            "Amount": "amount", "成交额": "amount", "金额": "amount"
                        }
                        df_local = df_local.rename(columns={c: col_map[c] for c in df_local.columns if c in col_map})

            # 统一数据清洗
            if df_local is not None and not df_local.empty:
                df_local["date"] = pd.to_datetime(df_local["date"])
                df_local = df_local.sort_values("date").reset_index(drop=True)
                # 确保所有必需列都存在
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    if col not in df_local.columns:
                        df_local[col] = 0
                    else:
                        df_local[col] = pd.to_numeric(df_local[col], errors='coerce')
                # 删除包含NaN的行
                df_local = df_local.dropna(subset=["open", "high", "low", "close", "volume"])
                # 如果amount列有NaN，用close * volume填充
                if df_local["amount"].isna().any():
                    df_local["amount"] = df_local["close"] * df_local["volume"]
                # 如果还有NaN，用0填充
                df_local = df_local.fillna(0)
                return df_local
        except Exception as e:
            if "数据需要更新" not in str(e):
                st.warning(f"本地数据加载失败: {e}，尝试在线获取...")

    # 2. 如果本地没有数据，尝试从多个数据源获取
    if df is None:
        data_sources = []

        # 2.1 尝试Akshare股票接口
        with st.spinner(f"正在从Akshare获取 {symbol} 的 {period} 数据..."):
            for attempt in range(1, max_retries + 1):
                try:
                    if period == "daily":
                        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
                    elif period == "30":
                        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period="30", adjust="qfq")

                    if df is not None and not df.empty:
                        st.success(f"✅ 从Akshare获取数据成功 ({len(df)} 条)")
                        break
                except Exception as e:
                    if attempt == max_retries:
                        st.warning(f"⚠️ Akshare获取失败: {e}")
                    time.sleep(1)

        # 2.2 如果是基金（ETF）且Akshare失败，尝试专用ETF接口
        is_etf = symbol.startswith('1') or symbol.startswith('5')
        if df is None or df.empty and is_etf:
            with st.spinner(f"正在从Akshare ETF专用接口获取 {symbol} 数据..."):
                try:
                    # fund_etf_hist_sina 专门用于ETF
                    # 需要带交易所前缀：sh或sz
                    prefix = "sh" if symbol.startswith('51') or symbol.startswith('50') else "sz"
                    etf_symbol = f"{prefix}{symbol}"

                    df = ak.fund_etf_hist_sina(symbol=etf_symbol)
                    if df is not None and not df.empty:
                        st.success(f"✅ 从Akshare ETF接口获取数据成功 ({len(df)} 条)")
                except Exception as e:
                    st.warning(f"⚠️ Akshare ETF接口获取失败: {e}")

        # 2.3 如果akshare都失败，尝试baostock
        if df is None or df.empty:
            with st.spinner(f"正在从Baostock获取 {symbol} 的 {period} 数据..."):
                try:
                    import baostock as bs

                    # Baostock QPS限制约20次/秒，添加延迟避免触发限制
                    time.sleep(0.1)  # 100ms延迟，确保不超过10次/秒

                    login_result = bs.login()
                    if login_result.error_code != '0':
                        st.warning(f"⚠️ Baostock登录失败: {login_result.error_msg}")
                        raise Exception(f"Baostock login failed: {login_result.error_msg}")

                    # 转换代码格式
                    if symbol.startswith('6'):
                        bs_code = f"sh.{symbol}"
                    elif symbol.startswith('0') or symbol.startswith('3'):
                        bs_code = f"sz.{symbol}"
                    else:
                        bs_code = f"sz.{symbol}"  # ETF默认sz

                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')

                    # 添加重试机制处理QPS限制
                    max_bs_retries = 3
                    for bs_attempt in range(max_bs_retries):
                        try:
                            rs = bs.query_history_k_data_plus(bs_code,
                                "date,open,high,low,close,volume,amount",
                                start_date=start_date, end_date=end_date,
                                frequency="d" if period == "daily" else "30",
                                adjustflag="3")  # 复权类型：3表示后复权

                            if rs.error_code != '0':
                                if '频率限制' in rs.error_msg or 'quota' in rs.error_msg.lower():
                                    if bs_attempt < max_bs_retries - 1:
                                        wait_time = (bs_attempt + 1) * 2
                                        st.info(f"⏳ 触发Baostock频率限制，等待{wait_time}秒后重试...")
                                        time.sleep(wait_time)
                                        continue
                                raise Exception(f"Query failed: {rs.error_msg}")

                            data_list = []
                            while (rs.error_code == '0') & rs.next():
                                data_list.append(rs.get_row_data())
                                # 每处理100行数据添加短暂延迟，避免连续请求过快
                                if len(data_list) % 100 == 0:
                                    time.sleep(0.05)

                            if data_list:
                                df = pd.DataFrame(data_list, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
                                # 转换数值类型
                                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                st.success(f"✅ 从Baostock获取数据成功 ({len(df)} 条)")
                            break

                        except Exception as bs_e:
                            if bs_attempt < max_bs_retries - 1 and ('频率' in str(bs_e) or 'quota' in str(bs_e).lower()):
                                continue
                            raise bs_e

                    bs.logout()
                except Exception as e:
                    st.warning(f"⚠️ Baostock获取失败: {e}")

        # 2.3 如果都失败，尝试Tushare（需要配置token）
        if df is None or df.empty:
            with st.spinner(f"正在从Tushare获取 {symbol} 的 {period} 数据..."):
                try:
                    import tushare as ts
                    # 尝试从环境变量或配置文件获取token
                    ts_token = os.environ.get('TUSHARE_TOKEN', '')
                    if ts_token:
                        pro = ts.pro_api(ts_token)

                        # 转换代码格式
                        if symbol.startswith('6'):
                            ts_code = f"{symbol}.SH"
                        else:
                            ts_code = f"{symbol}.SZ"

                        end_date = datetime.now().strftime('%Y%m%d')
                        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')

                        if period == "daily":
                            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                        else:
                            # Tushare的30分钟数据需要积分
                            df = pro.stk_mins(ts_code=ts_code, start_date=start_date, end_date=end_date, freq='30min')

                        if df is not None and not df.empty:
                            # 标准化列名
                            df = df.rename(columns={
                                'trade_date': 'date', 'open': 'open', 'high': 'high',
                                'low': 'low', 'close': 'close', 'vol': 'volume', 'amount': 'amount'
                            })
                            st.success(f"✅ 从Tushare获取数据成功 ({len(df)} 条)")
                    else:
                        st.info("ℹ️ Tushare未配置token，跳过")
                except Exception as e:
                    st.warning(f"⚠️ Tushare获取失败: {e}")

        if df is None or df.empty:
            st.error("❌ 所有数据源均无法获取数据")
            return None

        # Standardize columns
        rename_map = {
            "日期": "date", "时间": "date",
            "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount"
        }
        df.rename(columns=rename_map, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Clean numeric
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(",", "", regex=False)
                    .replace({"--": None, "": None})
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fix invalid data
        if "open" in df.columns:
            open_bad = (df["open"] == 0) | (df["open"].isna())
            if open_bad.any():
                df.loc[open_bad, "open"] = df["close"].shift(1)
                df["open"].fillna(df["close"], inplace=True)
        
        if "amount" not in df.columns or df["amount"].isna().all() or (df["amount"] == 0).all():
             df["amount"] = df["close"] * df["volume"]
            
        return df

def prepare_inputs(df, lookback, period="daily"):
    if len(df) < lookback:
        st.warning(f"数据长度 ({len(df)}) 小于回看周期 ({lookback})，将使用全部可用数据。")
        lookback = len(df)
        
    x_df = df.iloc[-lookback:][["open","high","low","close","volume","amount"]]
    x_timestamp = df.iloc[-lookback:]["date"]
    return x_df, pd.Series(x_timestamp)

def apply_price_limits(pred_df, last_close, limit_rate=0.1):
    pred_df = pred_df.reset_index(drop=True)
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")
    current_last_close = last_close
    
    for i in range(len(pred_df)):
        limit_up = current_last_close * (1 + limit_rate)
        limit_down = current_last_close * (1 - limit_rate)
        for col in cols:
            value = pred_df.at[i, col]
            if pd.notna(value):
                clipped = max(min(value, limit_up), limit_down)
                pred_df.at[i, col] = float(clipped)
        current_last_close = float(pred_df.at[i, "close"])
    return pred_df

def render_prediction_page():
    st.header("📈 Kronos 股价预测")
    st.markdown("基于 Transformer 的股价预测模型，支持日线与30分钟级别预测。")

    # 参数配置放在主内容区域，避免与侧边栏策略配置混淆
    st.header("🔮 预测配置")
    # Use session state to persist inputs across reruns
    # 默认值优先从 session_state 获取
    # 关键修改：如果主程序传递了 pred_symbol 且与当前 text_input 不一致，优先使用主程序传递的值
    # 这通过 session_state["pred_symbol"] 来实现联动
    # 但 streamlit 的 text_input 如果有 key，它的值会绑定到 session_state[key]
    # 所以我们需要在渲染 text_input 之前，确保 session_state[key] 是最新的

    # 强制同步：如果 session_state 中有 "pred_symbol"（来自主程序），则更新 input 的 key
    if "pred_symbol" in st.session_state and st.session_state["pred_symbol"] != st.session_state.get("pred_symbol_input"):
         st.session_state["pred_symbol_input"] = st.session_state["pred_symbol"]

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("股票代码 (预测)", key="pred_symbol_input")
    with col2:
        period = st.selectbox("数据周期", ["daily", "30"], format_func=lambda x: "日线" if x == "daily" else "30分钟", key="pred_period")

    # 反向同步：如果用户在这里修改了代码，也更新主程序的 pred_symbol
    st.session_state["pred_symbol"] = symbol

    with st.expander("高级模型参数", expanded=False):
        # 获取用户设置的默认值，如果不存在则使用初始默认值
        default_lookback = st.session_state.get("default_pred_lookback", 400)
        default_pred_len = st.session_state.get("default_pred_len", 30) # 根据用户图片，预测长度默认为30更合适
        default_temp = st.session_state.get("default_pred_temp", 0.9)   # 根据用户图片，温度默认为0.9
        default_top_p = st.session_state.get("default_pred_top_p", 0.9)

        col1, col2 = st.columns(2)
        with col1:
            lookback = st.slider("回看周期", 100, 500, default_lookback, 50, key="pred_lookback")
            pred_len = st.slider("预测长度", 10, 120, default_pred_len, 10, key="pred_len")
        with col2:
            temperature = st.slider("温度系数", 0.1, 2.0, default_temp, 0.1, help="数值越低越保守，越高越激进", key="pred_temp")
            top_p = st.slider("Top-P 采样", 0.1, 1.0, default_top_p, 0.1, key="pred_top_p")

        # 保存为默认值按钮
        if st.button("💾 保存为系统默认值"):
            st.session_state["default_pred_lookback"] = lookback
            st.session_state["default_pred_len"] = pred_len
            st.session_state["default_pred_temp"] = temperature
            st.session_state["default_pred_top_p"] = top_p
            st.success("已更新默认参数！")

    run_btn = st.button("🚀 开始预测", type="primary", key="pred_run_btn")

    # 添加强制刷新按钮
    col_refresh, _ = st.columns([1, 3])
    with col_refresh:
        force_refresh = st.button("🔄 强制重新预测", help="清除今日缓存，重新进行预测")

    # 检查是否需要自动运行
    # 我们使用一个 session 变量来存储上次预测的结果，避免重复计算
    # 只有当 auto_run_prediction 为 True 时，才重新计算

    # 缓存键：symbol + period + 参数哈希（确保参数变化时重新计算）
    param_hash = f"{lookback}_{pred_len}_{temperature}_{top_p}"
    cache_key = f"pred_result_{symbol}_{period}_{param_hash}"

    # 如果强制刷新，清除缓存
    if force_refresh:
        clear_cache_key = f"{cache_key}_date"
        if clear_cache_key in st.session_state:
            del st.session_state[clear_cache_key]
        st.info("已清除缓存，将重新进行预测")

    auto_run = False
    if st.session_state.get("auto_run_prediction", False):
        auto_run = True

    # 如果点击了按钮，强制运行
    should_run = run_btn or auto_run or force_refresh

    # 检查当天缓存（基于日期+参数）
    cache_valid = is_cache_valid(cache_key) and not force_refresh

    # 如果当天已有缓存结果，且没有强制运行请求，则直接使用缓存
    if not should_run and cache_valid:
        st.info(f"📦 显示今日已生成的预测结果 ({symbol} {period})")
        pred_df = get_cache(cache_key)
        should_run = False  # 确保不重新计算
        # 加载基础数据用于绘图（不强制更新）
        df = load_stock_data(symbol, period, force_update=False)
    elif not should_run and cache_key in st.session_state:
        # 有缓存但不是当天的
        st.info("显示历史预测结果 (参数未变，但数据可能不是今日最新)")
        pred_df = st.session_state[cache_key]
        should_run = False
        df = load_stock_data(symbol, period, force_update=False)
    else:
        # 进行新预测时，强制获取最新数据
        df = load_stock_data(symbol, period, force_update=True)

    # 主区域只显示图表
    if symbol and df is not None:
        # 如果是自动运行，显示加载状态
        if auto_run:
            st.info(f"正在自动为您生成 {symbol} 的股价预测...")

        if should_run:
            # 成功加载数据后，如果是自动运行，清除标记
            if auto_run:
                st.session_state["auto_run_prediction"] = False
            
            predictor = load_model()
            if predictor:
                # with st.spinner("正在生成预测... (首次运行可能需要下载模型)"): # spinner 会导致 UI 闪烁，自动模式下可以不需要
                status_ph = st.empty()
                status_ph.info("正在计算股价预测...")
                try:
                    x_df, x_timestamp = prepare_inputs(df, lookback, period)
                    last_date = df["date"].iloc[-1]
                    
                    if period == "daily":
                        y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)
                    else:
                        y_timestamp = [last_date + pd.Timedelta(minutes=30 * (i+1)) for i in range(pred_len)]
                        y_timestamp = pd.DatetimeIndex(y_timestamp)

                    pred_df = predictor.predict(
                        df=x_df, x_timestamp=x_timestamp, y_timestamp=pd.Series(y_timestamp),
                        pred_len=pred_len, T=temperature, top_p=top_p, sample_count=1, verbose=False
                    )
                    
                    pred_df["date"] = y_timestamp
                    pred_df = apply_price_limits(pred_df, df["close"].iloc[-1])
                    
                    # 存入缓存（带日期）
                    set_cache(cache_key, pred_df)
                    status_ph.empty()
                    
                except Exception as e:
                    status_ph.error(f"预测失败: {e}")
                    pred_df = None
            else:
                pred_df = None
        elif cache_key in st.session_state:
            pred_df = st.session_state[cache_key]
        else:
            pred_df = None

        if pred_df is not None:
            # Visualization
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # History (last 200)
            hist_subset = df.iloc[-200:].copy()
            fig.add_trace(go.Candlestick(
                x=hist_subset['date'], open=hist_subset['open'], high=hist_subset['high'],
                low=hist_subset['low'], close=hist_subset['close'], name="历史数据",
                increasing_line_color='red', decreasing_line_color='green',
                hovertext=[
                    f"日期: {d}<br>开盘: {o:.3f}<br>最高: {h:.3f}<br>最低: {l:.3f}<br>收盘: {c:.3f}"
                    for d, o, h, l, c in zip(hist_subset['date'], hist_subset['open'], hist_subset['high'], hist_subset['low'], hist_subset['close'])
                ],
                hoverinfo='text'
            ), row=1, col=1)

            # Prediction
            fig.add_trace(go.Candlestick(
                x=pred_df['date'], open=pred_df['open'], high=pred_df['high'],
                low=pred_df['low'], close=pred_df['close'], name="预测数据",
                increasing_line_color='orange', decreasing_line_color='cyan',
                hovertext=[
                    f"日期: {d}<br>开盘: {o:.3f}<br>最高: {h:.3f}<br>最低: {l:.3f}<br>收盘: {c:.3f} (预测)"
                    for d, o, h, l, c in zip(pred_df['date'], pred_df['open'], pred_df['high'], pred_df['low'], pred_df['close'])
                ],
                hoverinfo='text'
            ), row=1, col=1)
            
            # Volume
            colors_vol = ['red' if c >= o else 'green' for c, o in zip(hist_subset['close'], hist_subset['open'])]
            fig.add_trace(go.Bar(x=hist_subset['date'], y=hist_subset['volume'], marker_color=colors_vol, name="历史成交量"), row=2, col=1)
            
            colors_vol_pred = ['orange' if c >= o else 'cyan' for c, o in zip(pred_df['close'], pred_df['open'])]
            fig.add_trace(go.Bar(x=pred_df['date'], y=pred_df['volume'], marker_color=colors_vol_pred, name="预测成交量"), row=2, col=1)
            
            fig.update_layout(
                title=f"{symbol} 股价预测结果",
                height=600,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font=dict(color='white', size=12))
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("查看预测数据详情"):
                st.dataframe(pred_df)
                st.download_button("下载预测结果", pred_df.to_csv(index=False).encode('utf-8'), f"pred_{symbol}.csv", "text/csv")
        else:
             # Show simple chart preview only if not predicting
            fig_preview = go.Figure(data=[go.Candlestick(
                x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                hovertext=[
                    f"日期: {d}<br>开盘: {o:.3f}<br>最高: {h:.3f}<br>最低: {l:.3f}<br>收盘: {c:.3f}"
                    for d, o, h, l, c in zip(df['date'], df['open'], df['high'], df['low'], df['close'])
                ],
                hoverinfo='text'
            )])
            fig_preview.update_layout(
                title=f"{symbol} 历史走势",
                height=500,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font=dict(color='white', size=12))
            )
            st.plotly_chart(fig_preview, use_container_width=True)

    else:
        if symbol:
            st.error("无法获取数据")

# 暴露给主程序的后台运行函数
def run_prediction_background(symbol, period="daily"):
    """在后台静默运行预测并缓存结果"""
    if not symbol: return
    
    try:
        # Load model first
        if Kronos is None: return
        
        # Load tokenizer and model directly (not cached via st.cache_resource inside background task to avoid threading issues?)
        # Actually st.cache_resource works fine.
        predictor = load_model()
        if not predictor: return

        df = load_stock_data(symbol, period)
        if df is None: return
        
        # Params (use defaults from session state if available)
        lookback = st.session_state.get("default_pred_lookback", 400)
        pred_len = st.session_state.get("default_pred_len", 30)
        temperature = st.session_state.get("default_pred_temp", 0.9)
        top_p = st.session_state.get("default_pred_top_p", 0.9)
        
        x_df, x_timestamp = prepare_inputs(df, lookback, period)
        last_date = df["date"].iloc[-1]
        
        if period == "daily":
            y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)
        else:
            y_timestamp = [last_date + pd.Timedelta(minutes=30 * (i+1)) for i in range(pred_len)]
            y_timestamp = pd.DatetimeIndex(y_timestamp)

        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=pd.Series(y_timestamp),
            pred_len=pred_len, T=temperature, top_p=top_p, sample_count=1, verbose=False
        )
        
        pred_df["date"] = y_timestamp
        pred_df = apply_price_limits(pred_df, df["close"].iloc[-1])
        
        # Save to session state
        cache_key = f"pred_result_{symbol}_{period}"
        if "pred_result_cache" not in st.session_state:
             st.session_state["pred_result_cache"] = {}
        
        # 直接存入 session_state 顶层可能会在 rerun 时丢失？不，session_state 是持久的。
        st.session_state[cache_key] = pred_df
        # print(f"Background prediction finished for {symbol}")
        
    except Exception as e:
        print(f"Background prediction failed: {e}")
