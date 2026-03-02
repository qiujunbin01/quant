# OpenClaw Quant - 量化交易系统

基于双均线+RSRS时序策略的A股量化交易系统，支持ETF和个股实时监控、回测分析、AI股价预测。

## 功能特性

- **实时监控**: ETF/个股行情监控，K线图技术指标显示
- **回测分析**: 日线/30分钟双周期策略回测，收益对比
- **AI预测**: Kronos Transformer模型预测未来走势
- **综合研判**: 多维度AI分析（技术面+基本面+消息面）
- **智能选股**: AI自动挖掘热点题材和龙头股票

## 技术栈

- Python 3.8+
- Streamlit (Web界面)
- Pandas/NumPy (数据处理)
- Plotly (图表)
- Akshare/Baostock (数据源)
- PyTorch (AI预测模型)

## 安装部署

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/openclaw-quant.git
cd openclaw-quant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API密钥（可选）

编辑 `stock_analysis_module.py`，配置Kimi API密钥：

```python
API_KEY = "your-kimi-api-key"
```

### 4. 运行

```bash
streamlit run etf_monitor.py
```

## 主要模块

| 模块 | 说明 |
|------|------|
| `etf_monitor.py` | 主程序，Streamlit Web界面 |
| `backtest_final.py` | 回测引擎 |
| `stock_prediction_module.py` | AI股价预测（Kronos模型） |
| `stock_analysis_module.py` | AI综合研判 |
| `stock_selection_module.py` | 智能选股 |

## 策略说明

### 双均线+RSRS策略

- **高波动市场(ATR>1.5%)**: 使用MA10/MA20短周期组合
- **低波动市场(ATR≤1.5%)**: 使用MA20/MA60长周期组合
- **买入条件**: 价格>短均线>长均线，RSI在[40,70]，ROC20>0
- **卖出条件**: 价格跌破短均线或均线死叉，或RSI超买/超卖

### 手续费

- ETF: 佣金万1.2双向，免印花税
- 股票: 佣金万1.2双向 + 印花税万5（卖出）

## 数据源

- Akshare（主要）
- Baostock（备用）
- 本地CSV/Excel文件

## 免责声明

本系统仅供学习和研究使用，不构成投资建议。股市有风险，投资需谨慎。

## License

MIT License
