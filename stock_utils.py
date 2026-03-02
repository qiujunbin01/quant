
import pandas as pd
import akshare as ak
import os
from pypinyin import lazy_pinyin, Style
import json
import time

STOCK_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_list_cache.json")

def get_pinyin_initials(name):
    """
    Generate pinyin initials for a given Chinese string.
    e.g. "平安银行" -> "PAYH"
    """
    try:
        # pypinyin lazy_pinyin with FIRST_LETTER style
        # errors='ignore' prevents crashing on special chars
        initials = lazy_pinyin(name, style=Style.FIRST_LETTER, errors='ignore')
        return "".join(initials).upper()
    except:
        return ""

def update_stock_list_cache():
    """
    Fetch stock list from akshare and save to cache file.
    Returns list of dicts.
    """
    try:
        # Fetch A-share stock list
        # ak.stock_zh_a_spot_em() returns DataFrame with columns like '代码', '名称', etc.
        stock_list = []
        
        try:
            df_stock = ak.stock_zh_a_spot_em()
            for _, row in df_stock.iterrows():
                code = str(row['代码'])
                name = str(row['名称'])
                pinyin = get_pinyin_initials(name)
                stock_list.append({
                    "code": code,
                    "name": name,
                    "pinyin": pinyin,
                    "label": f"{code} {name} ({pinyin})"
                })
        except Exception as e:
            print(f"Error fetching stock list: {e}")

        # Fetch ETF list
        try:
            df_etf = ak.fund_etf_spot_em()
            # Columns might be '代码', '名称', ...
            for _, row in df_etf.iterrows():
                code = str(row['代码'])
                name = str(row['名称'])
                pinyin = get_pinyin_initials(name)
                stock_list.append({
                    "code": code,
                    "name": name,
                    "pinyin": pinyin,
                    "label": f"{code} {name} ({pinyin})"
                })
        except Exception as e:
            print(f"Error fetching ETF list: {e}")
            
        # Add some common ETFs manually if fetching failed or just to be safe
        # 159915 创业板
        # 510300 沪深300
        etfs = [
            ("159915", "创业板ETF"),
            ("510300", "沪深300ETF"),
            ("510050", "上证50ETF"),
            ("588000", "科创50ETF"),
        ]
        
        existing_codes = set(item['code'] for item in stock_list)
        
        for code, name in etfs:
            if code not in existing_codes:
                pinyin = get_pinyin_initials(name)
                stock_list.append({
                    "code": code,
                    "name": name,
                    "pinyin": pinyin,
                    "label": f"{code} {name} ({pinyin})"
                })

        # Save to json
        with open(STOCK_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump(stock_list, f, ensure_ascii=False)
            
        return stock_list
    except Exception as e:
        print(f"Error updating stock list: {e}")
        return []

def get_stock_options():
    """
    Get stock options list. Load from cache if exists.
    Otherwise fetch and update.
    """
    try:
        if os.path.exists(STOCK_LIST_FILE):
            # Load cache directly regardless of age (let user update manually if needed, or update periodically)
            # To avoid slow startup, we rely on cache.
            # But if cache is empty or invalid, we update.
            try:
                with open(STOCK_LIST_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data and len(data) > 100:
                        return data
            except:
                pass
        
        # If no cache or load failed, update
        return update_stock_list_cache()
    except:
        return []
