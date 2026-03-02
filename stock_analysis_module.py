
import streamlit as st
import requests
import json
import akshare as ak
import pandas as pd
from datetime import datetime

# Kimi2.5 API 配置
API_KEY = "sk-lFgCqGHVn3RsJDj5kypKIHl79sOP9YDTTqmEfi18ck17D6Ma"
BASE_URL = "https://api.moonshot.cn/v1"  # Kimi API地址
# Kimi模型: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k 或 kimi-latest

def get_stock_news(symbol, limit=5):
    """获取个股新闻"""
    try:
        # akshare 的 symbol 格式通常是 6 位数字
        # stock_news_em 接口需要 6 位代码
        clean_symbol = symbol
        if "." in symbol:
            clean_symbol = symbol.split(".")[0]
        elif len(symbol) > 6: # handle sh600519
            clean_symbol = symbol[-6:]
            
        news_df = ak.stock_news_em(symbol=clean_symbol)
        if news_df is not None and not news_df.empty:
            # 选取最近的 limit 条
            recent_news = news_df.head(limit)
            news_list = []
            for _, row in recent_news.iterrows():
                title = row.get('新闻标题', '无标题')
                content = row.get('新闻内容', '')
                date = row.get('发布时间', '')
                news_list.append(f"- [{date}] {title}")
            return "\n".join(news_list)
    except Exception as e:
        print(f"获取新闻失败: {e}")
    return "暂无最新新闻数据。"

def get_stock_finance(symbol):
    """获取个股财务摘要"""
    try:
        clean_symbol = symbol
        if len(symbol) > 6:
            clean_symbol = symbol[-6:]
            
        # 财务摘要
        df = ak.stock_financial_abstract(symbol=clean_symbol)
        if df is not None and not df.empty:
            # 选取最近一期
            latest = df.iloc[0]
            # 假设列名包含特定财务指标
            # akshare 返回的列名可能变化，这里做简单处理
            return latest.to_dict()
    except Exception as e:
        print(f"获取财务数据失败: {e}")
    return "暂无财务摘要数据。"

def call_kimi_chat(messages, model="kimi-latest"):
    """调用 Kimi2.5 API 进行对话 (支持联网搜索)"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.3
        # 注意：Kimi API不支持web_search工具，联网功能需要在提示词中引导
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=data, timeout=120)

        if response.status_code == 200:
            try:
                res_json = response.json()
                return res_json['choices'][0]['message']['content']
            except json.JSONDecodeError as e:
                return f"JSON解析失败: {e}"
        elif response.status_code == 401:
            return "API认证失败：API Key无效或已过期，请检查API Key配置"
        else:
            return f"API 调用失败: {response.status_code} - {response.text[:200]}"
    except requests.exceptions.Timeout:
        return "请求超时：API响应时间过长，请稍后重试"
    except Exception as e:
        return f"请求发生错误: {e}"

def generate_analysis_report(symbol, current_price, signal_info, prediction_info, news_context, finance_context):
    """生成综合研判报告"""
    
    # 提取技术面信号（可能包含日线和30分钟）
    tech_signals = ""
    if isinstance(signal_info, dict):
        if "signal" in signal_info: # 旧格式或单周期
             tech_signals += f"- 日线/当前周期信号：{signal_info.get('signal', '未知')} ({signal_info.get('reason', '')})\n"
        if "daily_signal" in signal_info:
             tech_signals += f"- 日线级别信号：{signal_info['daily_signal'].get('signal', '未知')} ({signal_info['daily_signal'].get('reason', '')})\n"
        if "min30_signal" in signal_info:
             tech_signals += f"- 30分钟级别信号：{signal_info['min30_signal'].get('signal', '未知')} ({signal_info['min30_signal'].get('reason', '')})\n"
    else:
        tech_signals = str(signal_info)

    system_prompt = """你是一名拥有20年经验的资深金融分析师和宏观经济学家。
你的任务是根据提供的多维度数据（技术面、AI预测、基本面、消息面），对股票进行深度综合研判。
请保持客观、理性，并给出明确的操作建议（买入、卖出、持有、观望）。
报告风格应专业、条理清晰，使用 Markdown 格式。
特别注意：技术面分析需涵盖【日线级别】（大趋势）和【30分钟级别】（短线买卖点），并分析两者的共振或背离情况。

你可以使用联网搜索功能获取最新的市场新闻、公司公告和行业动态，以补充分析。
"""

    user_prompt = f"""
请对股票代码：{symbol} 进行综合分析。

【1. 市场数据】
- 当前价格：{current_price}
- 技术面信号汇总：
{tech_signals}

【2. AI 模型预测 (未来30天)】
- 预测趋势：{prediction_info}

【3. 公司基本面与财务】
{finance_context}

【4. 近期新闻舆情】
{news_context}

【分析要求】
请结合当前国家宏观政策（如货币政策、行业扶持政策等）和整体经济形势，撰写一份分析报告：
1. **核心观点**：一句话总结当前投资价值。
2. **多维解读**：
   - **技术面深度分析**：
     - **日线级别**：分析长期趋势、主力资金动向及关键支撑/压力位。
     - **30分钟级别**：分析短线买卖点、超买超卖情况及背离信号。
     - **周期共振**：判断日线与30分钟信号是否一致（共振增强信心，背离提示风险）。
   - **基本面分析**：财务状况与估值水平。
   - **消息面与政策面**：结合新闻和宏观背景。
3. **风险提示**：列出潜在的下行风险。
4. **操作建议**：针对短线和中长线投资者分别给出建议。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return call_kimi_chat(messages)

def render_analysis_page(symbol, current_price, signal_data, pred_df=None):
    st.header("🤖 AI 综合研判")
    st.info("本模块利用 DeepSeek 大模型，结合技术指标、AI预测数据、实时新闻及宏观政策，为您生成深度投资报告。")
    
    if st.button("🚀 生成综合研判报告", type="primary", use_container_width=True):
        with st.spinner("正在搜集新闻、财务数据并进行深度思考... (预计耗时 10-20 秒)"):
            # 1. 准备数据
            # 预测数据摘要
            pred_info = "暂无预测数据"
            pred_trend_desc = ""
            if pred_df is not None and not pred_df.empty:
                start_p = pred_df.iloc[0]['close']
                end_p = pred_df.iloc[-1]['close']
                change = (end_p - start_p) / start_p * 100
                trend = "上涨" if change > 0 else "下跌"
                pred_trend_desc = f"模型预测未来30个周期将{trend}约 {abs(change):.2f}% (起始 {start_p:.2f} -> 期末 {end_p:.2f})"
                pred_info = pred_trend_desc
            
            # 2. 获取外部数据
            news = get_stock_news(symbol)
            finance = get_stock_finance(symbol)
            
            # 3. 调用 LLM
            report = generate_analysis_report(
                symbol=symbol,
                current_price=current_price,
                signal_info=signal_data,
                prediction_info=pred_info,
                news_context=news,
                finance_context=str(finance)
            )
            
            # 4. 展示结果
            st.markdown("### 📄 深度分析报告")
            st.markdown(report)
            
            # 5. 可视化：插入预测K线图（如果有）
            if pred_df is not None and not pred_df.empty:
                st.subheader("🔮 AI 预测走势图")
                import plotly.graph_objects as go
                fig_pred = go.Figure()
                
                # 尝试获取历史数据以补充前15天走势
                hist_days = 15
                hist_data = None
                try:
                    # 复用 akshare 获取历史日线
                    start_date_hist = (datetime.now() - pd.Timedelta(days=hist_days*2)).strftime("%Y%m%d") # 多取一点防休市
                    hist_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date_hist, adjust="qfq")
                    if hist_df is not None and not hist_df.empty:
                         col_map = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low"}
                         hist_df = hist_df.rename(columns={c: col_map[c] for c in hist_df.columns if c in col_map})
                         hist_df['date'] = pd.to_datetime(hist_df['date'])
                         hist_data = hist_df.tail(hist_days)
                except:
                    pass
                
                # 绘制历史K线 (前15天)
                if hist_data is not None and not hist_data.empty:
                     fig_pred.add_trace(go.Candlestick(
                        x=hist_data['date'],
                        open=hist_data['open'],
                        high=hist_data['high'],
                        low=hist_data['low'],
                        close=hist_data['close'],
                        name='历史走势 (近15天)',
                        increasing_line_color='red',
                        decreasing_line_color='green',
                        hovertext=[
                            f"日期: {d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else d}<br>"
                            f"开盘: {o:.3f}<br>最高: {h:.3f}<br>最低: {l:.3f}<br>收盘: {c:.3f}"
                            for d, o, h, l, c in zip(hist_data['date'], hist_data['open'], hist_data['high'], hist_data['low'], hist_data['close'])
                        ],
                        hoverinfo='text'
                    ))

                # 绘制预测K线 (未来30天)
                # 检查预测数据是否包含OHLC列
                has_ohlc = all(col in pred_df.columns for col in ['open', 'high', 'low'])

                if has_ohlc:
                    # 有完整OHLC数据，使用Candlestick
                    fig_pred.add_trace(go.Candlestick(
                        x=pred_df['date'],
                        open=pred_df['open'],
                        high=pred_df['high'],
                        low=pred_df['low'],
                        close=pred_df['close'],
                        name='AI预测 (未来30天)',
                        increasing_line_color='orange',
                        decreasing_line_color='cyan',
                        hovertext=[
                            f"日期: {d}<br>开盘: {o:.3f}<br>最高: {h:.3f}<br>最低: {l:.3f}<br>收盘: {c:.3f} (AI预测)"
                            for d, o, h, l, c in zip(pred_df['date'], pred_df['open'], pred_df['high'], pred_df['low'], pred_df['close'])
                        ],
                        hoverinfo='text'
                    ))
                else:
                    # 只有收盘价，使用折线
                    fig_pred.add_trace(go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['close'],
                        mode='lines+markers',
                        name='AI预测 (未来30天)',
                        line=dict(color='cyan', width=2, dash='dash')
                    ))
                
                # 添加分割线
                if hist_data is not None and not hist_data.empty:
                    last_hist_date = hist_data['date'].iloc[-1]
                    
                    # Plotly 的 add_vline 在某些版本中处理日期轴上的位置时，如果是字符串或 Timestamp，可能会在计算 annotation 位置时出错
                    # 尤其是当它试图计算 label 位置 (eX = _mean(X)) 时
                    # 最稳妥的方法是使用毫秒时间戳（对于 Date 轴）
                    
                    vline_x = last_hist_date
                    if isinstance(last_hist_date, pd.Timestamp):
                        # 转换为毫秒时间戳
                        vline_x = last_hist_date.timestamp() * 1000
                    elif isinstance(last_hist_date, str):
                        try:
                            ts = pd.to_datetime(last_hist_date)
                            vline_x = ts.timestamp() * 1000
                        except:
                            pass
                            
                    # 注意：如果轴是 category 类型（通常不是），则需要用索引。
                    # 这里我们假设是 Date 轴。
                    
                    # 为了避免 annotation 计算位置时的类型错误，我们可以不使用 add_vline 的 annotation 参数，
                    # 而是分开画线和添加注释
                    
                    # 1. 画线
                    fig_pred.add_shape(
                        type="line",
                        x0=vline_x, y0=0, x1=vline_x, y1=1,
                        xref="x", yref="paper",
                        line=dict(color="white", width=1, dash="dash")
                    )
                    
                    # 2. 添加注释 (手动控制位置)
                    # 为了避免 y 轴坐标问题，yref="paper"
                    fig_pred.add_annotation(
                        x=vline_x, y=0.05,
                        xref="x", yref="paper",
                        text="预测起点",
                        showarrow=False,
                        font=dict(color="white"),
                        bgcolor="black",
                        opacity=0.8
                    )

                fig_pred.update_layout(
                    title=f"{symbol} 走势回顾与预测 ({pred_trend_desc})",
                    xaxis_title="日期",
                    yaxis_title="价格",
                    template="plotly_dark",
                    height=450,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified',
                    hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font=dict(color='white', size=12))
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            # 6. 下载报告按钮
            # 将报告内容转换为 PDF 需要额外库（如 fpdf, reportlab），且 Streamlit Cloud 环境可能缺字库。
            # 这里为了通用性，我们先提供 Markdown/Text 下载，或者尝试生成 HTML 下载（浏览器可转PDF）
            # 生成带样式的 HTML 报告
            html_report = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #2E86C1; }}
                    .highlight {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>{symbol} 深度投资研判报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <div class="highlight">
                    <p><strong>当前价格:</strong> {current_price}</p>
                    <p><strong>AI预测:</strong> {pred_trend_desc}</p>
                </div>
                <hr>
                {report.replace(chr(10), '<br>')}
                <hr>
                <p><em>本报告由 AI 智能量化决策系统自动生成，仅供参考，不构成投资建议。</em></p>
            </body>
            </html>
            """
            
            st.download_button(
                label="📥 下载报告 (HTML格式，可打印为PDF)",
                data=html_report,
                file_name=f"{symbol}_analysis_report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )
            
            # 7. 补充展示原始数据
            with st.expander("查看参考源数据"):
                st.subheader("新闻数据")
                st.text(news)
                st.subheader("财务摘要")
                st.write(finance)

