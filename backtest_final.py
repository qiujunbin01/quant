"""
创业板ETF（159915）量化交易策略回测系统
策略：双均线 + RSRS择时 + 波动率控制
佣金：0.012%（单边），印花税：0%
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_and_process_data():
    """加载并处理数据"""
    print("="*60)
    print(" 创业板ETF量化交易策略回测系统")
    print("="*60)
    print("\n正在加载数据...")
    
    # 读取原始分时数据
    data_file = r"C:\Users\Administrator\Desktop\openclaw\量化交易\Table159915-20250627.xlsx"
    df = pd.read_excel(data_file)
    
    # 解析日期
    df['Date'] = df['时间'].str.split(',').str[0]
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 清洗数值列
    numeric_cols = ['开盘', '最高', '最低', '收盘']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 转换为日线数据
    daily = df.groupby('Date').agg({
        '开盘': 'first',
        '最高': 'max',
        '最低': 'min', 
        '收盘': 'last',
        '总手': 'sum',
        '金额': 'sum'
    }).reset_index()
    
    daily = daily.sort_values('Date').reset_index(drop=True)
    
    print(f"数据加载成功，共 {len(daily)} 个交易日")
    print(f"时间范围: {daily['Date'].min().strftime('%Y-%m-%d')} 至 {daily['Date'].max().strftime('%Y-%m-%d')}")
    
    return daily


def calculate_indicators(df):
    """计算技术指标"""
    print("\n正在计算技术指标...")
    
    ma_short = 5
    ma_long = 20
    rsrs_n = 18
    rsrs_m = 600
    
    # 移动平均线
    df['MA5'] = df['收盘'].rolling(window=ma_short).mean()
    df['MA20'] = df['收盘'].rolling(window=ma_long).mean()
    
    # 均线交叉信号
    df['MA5_above_MA20'] = df['MA5'] > df['MA20']
    df['MA5_above_MA20_prev'] = df['MA5_above_MA20'].shift(1)
    df['Golden_Cross'] = (df['MA5_above_MA20'] == True) & (df['MA5_above_MA20_prev'] == False)
    df['Death_Cross'] = (df['MA5_above_MA20'] == False) & (df['MA5_above_MA20_prev'] == True)
    
    # RSI指标
    delta = df['收盘'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 波动率
    df['Returns'] = df['收盘'].pct_change()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    df['Volatility_60_Median'] = df['Volatility_20'].rolling(window=60).median()
    
    # 成交量均线
    df['Volume_MA20'] = df['总手'].rolling(window=20).mean()
    
    # RSRS指标
    print("计算RSRS指标...")
    rsrs_slope = []
    for i in range(len(df)):
        if i < rsrs_n:
            rsrs_slope.append(np.nan)
        else:
            window = df.iloc[i-rsrs_n+1:i+1]
            high_low = window['最高'] - window['最低'].shift(1)
            high_low = high_low.dropna()
            if len(high_low) < rsrs_n - 1:
                rsrs_slope.append(np.nan)
                continue
            x = np.arange(len(high_low))
            y = high_low.values
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                rsrs_slope.append(np.nan)
                continue
            try:
                slope = np.polyfit(x, y, 1)[0]
                rsrs_slope.append(slope)
            except:
                rsrs_slope.append(np.nan)
    
    df['RSRS_Slope'] = rsrs_slope
    
    # RSRS Z分数
    zscore = []
    for i in range(len(df)):
        if i < rsrs_m:
            zscore.append(np.nan)
        else:
            slope_series = df['RSRS_Slope'].iloc[i-rsrs_m:i].dropna()
            if len(slope_series) < rsrs_m * 0.5:
                zscore.append(np.nan)
                continue
            current_slope = df.loc[i, 'RSRS_Slope']
            if np.isnan(current_slope):
                zscore.append(np.nan)
                continue
            mean = slope_series.mean()
            std = slope_series.std()
            if std == 0 or np.isnan(std):
                zscore.append(np.nan)
                continue
            z = (current_slope - mean) / std
            zscore.append(z)
    
    df['RSRS_ZScore'] = zscore
    
    print("技术指标计算完成")
    return df


def run_backtest(df):
    """运行回测"""
    print("\n" + "="*50)
    print("开始回测...")
    print("="*50)
    
    # 策略参数
    commission = 0.00012  # 0.012% 佣金
    stop_loss = 0.08      # 固定止损 8%
    profit_take_3 = 0.30   # 止盈30%
    trailing_stop = 0.08    # 移动止盈回撤 8%
    trailing_trigger = 0.15 # 移动止盈触发点 15%
    max_hold_days = 40     # 最大持仓天数
    base_position = 0.25   # 基础仓位 25%
    max_position = 0.60     # 最大仓位 60%
    
    # 初始化
    cash = 1000000  # 初始资金100万
    position = 0    
    entry_price = 0  
    entry_date = None  
    hold_days = 0  
    peak_price = 0  
    stop_loss_price = 0  
    
    initial_cash = cash
    warmup = 600  # 预热期
    
    trades = []
    equity_curve = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        current_price = row['收盘']
        current_date = row['Date']
        
        if idx < warmup:
            equity_curve.append({
                'Date': current_date,
                'Equity': cash + position * current_price,
                'Position': 0,
            })
            continue
        
        # 出场信号判断
        if position > 0:
            hold_days += 1
            if current_price > peak_price:
                peak_price = current_price
            
            exit_signal = False
            exit_reason = ""
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 1. 固定止损
            if current_price <= stop_loss_price:
                exit_signal = True
                exit_reason = "止损"
            
            # 2. 止盈30%
            if not exit_signal and pnl_pct >= profit_take_3:
                exit_signal = True
                exit_reason = "止盈30%"
            
            # 3. 移动止盈
            if not exit_signal and pnl_pct >= trailing_trigger:
                if peak_price > 0 and (peak_price - current_price) / peak_price >= trailing_stop:
                    exit_signal = True
                    exit_reason = "移动止盈"
            
            # 4. 均线死叉
            if not exit_signal and row['Death_Cross']:
                exit_signal = True
                exit_reason = "死叉"
            
            # 5. 时间止盈
            if not exit_signal and hold_days >= max_hold_days:
                exit_signal = True
                exit_reason = "超时"
            
            # 执行出场
            if exit_signal:
                sell_value = position * current_price * (1 - commission)
                cash = cash + sell_value
                
                trades.append({
                    '入场日期': entry_date.strftime('%Y-%m-%d'),
                    '入场价格': round(entry_price, 4),
                    '出场日期': current_date.strftime('%Y-%m-%d'),
                    '出场价格': round(current_price, 4),
                    '持仓天数': hold_days,
                    '收益率(%)': round(pnl_pct * 100, 2),
                    '出场原因': exit_reason
                })
                
                position = 0
                entry_price = 0
                entry_date = None
                hold_days = 0
                peak_price = 0
                stop_loss_price = 0
        
        # 入场信号判断
        if position == 0:
            # 必要条件
            ma_condition = row['Golden_Cross']
            volume_condition = row['总手'] >= row['Volume_MA20'] * 0.8
            # 价格条件已移除：原条件(current_price <= highest * 0.95)过于严格
            # 在日线数据中，收盘价接近最高价是正常现象，此条件导致几乎没有交易机会

            # 辅助条件
            rsrs_condition = row['RSRS_ZScore'] > -0.5 if not pd.isna(row['RSRS_ZScore']) else True
            volatility_condition = True
            if not pd.isna(row['Volatility_20']) and not pd.isna(row['Volatility_60_Median']) and row['Volatility_60_Median'] > 0:
                volatility_condition = row['Volatility_20'] < row['Volatility_60_Median']

            essential_ok = ma_condition and volume_condition
            aux_count = sum([rsrs_condition, volatility_condition])
            
            if essential_ok and aux_count >= 1:
                # 计算仓位
                vol_ratio = 1.0
                if not pd.isna(row['Volatility_20']) and not pd.isna(row['Volatility_60_Median']) and row['Volatility_60_Median'] > 0:
                    vol_ratio = row['Volatility_20'] / row['Volatility_60_Median']
                
                if vol_ratio > 1.2:
                    position_ratio = base_position * 0.6
                elif vol_ratio < 0.8:
                    position_ratio = base_position * 1.2
                else:
                    position_ratio = base_position
                
                position_ratio = min(position_ratio, max_position)
                
                # 买入
                buy_amount = cash * position_ratio
                shares = int(buy_amount / current_price / 100) * 100
                
                if shares > 0:
                    actual_buy = shares * current_price
                    comm = actual_buy * commission

                    cash = cash - actual_buy - comm
                    position = shares  # BUG修复：设置持仓数量
                    entry_price = current_price
                    entry_date = current_date
                    hold_days = 0
                    peak_price = current_price
                    stop_loss_price = entry_price * (1 - stop_loss)
        
        # 记录权益
        equity = cash + position * current_price
        equity_curve.append({
            'Date': current_date,
            'Equity': round(equity, 2),
            'Position': position,
        })
    
    # 最终平仓
    if position > 0:
        last_row = df.iloc[-1]
        sell_value = position * last_row['收盘'] * (1 - commission)
        cash = cash + sell_value
        
        pnl_pct = (last_row['收盘'] - entry_price) / entry_price
        trades.append({
            '入场日期': entry_date.strftime('%Y-%m-%d'),
            '入场价格': round(entry_price, 4),
            '出场日期': last_row['Date'].strftime('%Y-%m-%d'),
            '出场价格': round(last_row['收盘'], 4),
            '持仓天数': hold_days,
            '收益率(%)': round(pnl_pct * 100, 2),
            '出场原因': '策略结束'
        })
    
    # 计算绩效
    calculate_performance(equity_curve, trades, initial_cash)
    
    return equity_curve, trades


def calculate_performance(equity_curve, trades, initial_cash):
    """计算并显示绩效指标"""
    print("\n" + "="*60)
    print(" 回测结果汇总 ")
    print("="*60)
    
    equity_df = pd.DataFrame(equity_curve)
    
    final_equity = equity_df['Equity'].iloc[-1]
    total_return = (final_equity - initial_cash) / initial_cash
    
    trading_days = len(equity_df)
    years = trading_days / 252
    annual_return = (final_equity / initial_cash) ** (1/years) - 1
    
    equity_df['Returns'] = equity_df['Equity'].pct_change()
    volatility = equity_df['Returns'].std() * np.sqrt(252)
    
    risk_free_rate = 0.03
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    equity_df['Cummax'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Cummax'] - equity_df['Equity']) / equity_df['Cummax']
    max_drawdown = equity_df['Drawdown'].max()
    avg_drawdown = equity_df['Drawdown'].mean()
    
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        win_trades = trades_df[trades_df['收益率(%)'] > 0]
        loss_trades = trades_df[trades_df['收益率(%)'] <= 0]
        
        win_rate = len(win_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = win_trades['收益率(%)'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['收益率(%)'].mean() if len(loss_trades) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        profit_count = len(win_trades)
        loss_count = len(loss_trades)
    else:
        win_rate = avg_win = avg_loss = profit_loss_ratio = 0
        profit_count = loss_count = 0
    
    print(f"\n回测期间: {equity_df['Date'].iloc[0].strftime('%Y-%m-%d')} 至 {equity_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"交易天数: {trading_days} 天 ({years:.1f} 年)")
    print("-" * 60)
    print(f"初始资金:     {initial_cash:>15,.2f} 元")
    print(f"最终权益:     {final_equity:>15,.2f} 元")
    print(f"总收益率:     {total_return*100:>14.2f}%")
    print(f"年化收益率:   {annual_return*100:>14.2f}%")
    print("-" * 60)
    print(f"年化波动率:   {volatility*100:>14.2f}%")
    print(f"夏普比率:     {sharpe_ratio:>15.3f}")
    print(f"最大回撤:      {max_drawdown*100:>14.2f}%")
    print(f"平均回撤:      {avg_drawdown*100:>14.2f}%")
    print("-" * 60)
    print(f"交易次数:     {len(trades_df):>15} 次")
    print(f"盈利次数:     {profit_count:>15} 次")
    print(f"亏损次数:     {loss_count:>15} 次")
    print(f"胜率:         {win_rate:>14.1f}%")
    print(f"平均盈利:     {avg_win:>14.2f}%")
    print(f"平均亏损:     {avg_loss:>14.2f}%")
    print(f"盈亏比:       {profit_loss_ratio:>14.2f}")
    
    print("\n" + "-" * 60)
    print(" 交易明细 ")
    print("-" * 60)
    if len(trades_df) > 0:
        for i, row in trades_df.iterrows():
            print(f"{row['入场日期']} @ {row['入场价格']:.3f} → {row['出场日期']} @ {row['出场价格']:.3f} | {row['收益率(%)']:+.2f}% | {row['持仓天数']}天 | {row['出场原因']}")
    
    # 保存结果
    equity_df.to_csv('equity_curve.csv', index=False, encoding='utf-8-sig')
    trades_df.to_csv('trade_records.csv', index=False, encoding='utf-8-sig')
    
    print("\n结果已保存至 equity_curve.csv 和 trade_records.csv")


def main():
    """主函数"""
    # 加载数据
    daily_df = load_and_process_data()
    
    # 计算指标
    daily_df = calculate_indicators(daily_df)
    
    # 运行回测
    equity_curve, trades = run_backtest(daily_df)
    
    return equity_curve, trades


if __name__ == "__main__":
    results = main()
