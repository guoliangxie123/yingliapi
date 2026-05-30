import os
import re
import base64
import json
import requests
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.express as px
from longport.openapi import Config, QuoteContext

# ==========================================
# 1. 页面基本配置与初始化缓存
# ==========================================
st.set_page_config(
    page_title="Sell Put 智能识别与风险监控舱",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Sell Put 智能识别与实时风险监控舱")
st.markdown("结合 **Gemini 3.5 多模态 OCR** 与 **LongPort 实时行情**，全自动进行穿透式名义价值、希腊字母与年化收益监控。")

# 初始化 Session State（避免每次点按钮都重新执行 OCR）
if "raw_options" not in st.session_state:
    st.session_state.raw_options = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""

# ==========================================
# 2. 侧边栏配置区 ( 读取云端 Secrets 配置 )
# ==========================================
st.sidebar.header("⚙️ 1. Gemini API 配置")
default_gemini_key = st.secrets.get("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input("Gemini API Key", value=default_gemini_key, type="password")

model_options = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
model_selected = st.sidebar.selectbox("选择分析模型", options=model_options, index=0)

use_proxy = st.sidebar.checkbox("启用网络代理 (Gemini)", value=False)
proxy_url = st.sidebar.text_input("代理服务器地址", value="http://127.0.0.1:1601", disabled=not use_proxy)
gemini_proxies = {"http": proxy_url, "https": proxy_url} if use_proxy and proxy_url else None

st.sidebar.divider()
st.sidebar.header("🔌 2. LongPort API 配置")
default_lp_app_key = st.secrets.get("LONGPORT_APP_KEY", "")
default_lp_app_secret = st.secrets.get("LONGPORT_APP_SECRET", "")
default_lp_access_token = st.secrets.get("LONGPORT_ACCESS_TOKEN", "")

lp_app_key = st.sidebar.text_input("APP KEY", value=default_lp_app_key, type="password")
lp_app_secret = st.sidebar.text_input("APP SECRET", value=default_lp_app_secret, type="password")
lp_access_token = st.sidebar.text_input("ACCESS TOKEN", value=default_lp_access_token, type="password")

# ==========================================
# 3. LongPort 资源连接池管理
# ==========================================
@st.cache_resource
def get_longport_config(app_key, app_secret, access_token):
    if not (app_key and app_secret and access_token): return None
    try: return Config(app_key=app_key, app_secret=app_secret, access_token=access_token)
    except Exception: return None

@st.cache_resource
def get_quote_context(app_key, app_secret, access_token):
    cfg = get_longport_config(app_key, app_secret, access_token)
    if not cfg: return None
    try: return QuoteContext(cfg)
    except Exception: return None

quote_ctx = get_quote_context(lp_app_key, lp_app_secret, lp_access_token)
if quote_ctx:
    st.sidebar.success("✅ LongPort 实时行情连接已就绪")
else:
    st.sidebar.warning("⚠️ 行情未连接。将默认使用 OCR 提取的静态价格。")

# ==========================================
# 4. 辅助函数
# ==========================================
def format_market_symbol(symbol):
    symbol = symbol.strip().upper()
    if "." in symbol: return symbol
    return f"{symbol.zfill(5)}.HK" if re.match(r"^\d+$", symbol) else f"{symbol}.US"

def build_longport_option_symbol(underlying, expiry, strike, option_type="P"):
    try:
        underlying = underlying.split(".")[0].upper()
        clean_expiry = expiry.replace("-", "").replace("/", "")
        if len(clean_expiry) == 8: clean_expiry = clean_expiry[2:]
        if len(clean_expiry) != 6: return None
        strike_str = f"{int(float(strike) * 1000):08d}"
        suffix = "US" if not re.match(r"^\d+$", underlying) else "HK"
        return f"{underlying}{clean_expiry}{option_type}{strike_str}.{suffix}"
    except Exception: return None

def encode_uploaded_file(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8'), uploaded_file.type

def get_action_suggestion(profit_pct, buffer_pct, dte, delta):
    """根据核心指标生成智能操作建议"""
    if profit_pct >= 60 and dte > 15: return "🎯 买入平仓 (BTC)，提早落袋"
    if profit_pct >= 80: return "🎯 收益见顶，建议平仓"
    if buffer_pct < 0 and dte <= 7: return "⚠️ 准备展期 (Roll Out) 或接盘正股"
    if profit_pct <= -50 and buffer_pct > 5: return "⏳ 护城河尚在，吃时间价值"
    if dte <= 3 and buffer_pct > 3: return "🤑 躺平等归零"
    return "👀 持续观望"

# ==========================================
# 5. Gemini OCR & LongPort 行情逻辑
# ==========================================
def extract_put_options_from_image(base64_data, mime_type, api_key, model_name, proxies):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    schema = {
        "type": "OBJECT",
        "properties": {
            "put_options": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "underlying_name": {"type": "STRING"},
                        "strike_price": {"type": "STRING"},
                        "expiration_date": {"type": "STRING"},
                        "quantity": {"type": "STRING"},
                        "current_price": {"type": "STRING"},
                        "cost_price": {"type": "STRING"}
                    },
                    "required": ["underlying_name", "strike_price", "expiration_date", "quantity", "current_price", "cost_price"]
                }
            }
        },
        "required": ["put_options"]
    }
    payload = {
        "contents": [{"parts": [{"text": "提取图片中所有的 Put 期权（认沽/沽/代码含 P）。严格提取标的代码、行权价、到期日 (YYYY-MM-DD)、持仓数量 (Sell Put 为负数 )、现价和成本价。排除 Call 和正股。"}, {"inlineData": {"mimeType": mime_type, "data": base64_data}}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}
    }
    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, proxies=proxies, timeout=30)
    if resp.status_code == 200:
        return json.loads(resp.json()['candidates'][0]['content']['parts'][0]['text'])
    raise RuntimeError(f"OCR 失败: {resp.text}")

def get_realtime_quotes(symbols, is_option=False):
    """增强版：期权行情额外提取 IV 和 Delta"""
    if not quote_ctx or not symbols: return {}
    try:
        clean_symbols = list(set([s for s in symbols if s and s != "UNKNOWN"]))
        if not clean_symbols: return {}
        if is_option:
            quotes = quote_ctx.option_quote(clean_symbols)
            return {q.symbol: {
                "price": float(q.last_done),
                "iv": float(getattr(q, 'implied_volatility', 0.0) or 0.0) * 100,
                "delta": float(getattr(q, 'delta', 0.0) or 0.0)
            } for q in quotes}
        else:
            quotes = quote_ctx.quote(clean_symbols)
            return {q.symbol: float(q.last_done) for q in quotes}
    except Exception as e:
        st.warning(f"拉取行情异常: {e}")
        return {}

# ==========================================
# 6. 主页面交互与数据处理
# ==========================================
col1, col2 = st.columns([1, 1.8])

with col1:
    st.subheader("📸 1. 数据录入区")
    uploaded_file = st.file_uploader("支持 PNG, JPG 等格式持仓截图", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)
        # 如果上传了新图片，清空旧缓存
        if uploaded_file.name != st.session_state.uploaded_filename:
            st.session_state.raw_options = []
            st.session_state.uploaded_filename = uploaded_file.name

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        run_ocr = st.button("🚀 开始智能 OCR 分析", type="primary", use_container_width=True, disabled=(uploaded_file is None))
    with c_btn2:
        refresh_quotes = st.button("🔄 仅刷新实时行情", use_container_width=True, disabled=(len(st.session_state.raw_options) == 0))

# 核心处理逻辑触发
if run_ocr and uploaded_file and api_key_input:
    with st.spinner("正在调用 Gemini 多模态模型解析截图..."):
        try:
            b64, mtype = encode_uploaded_file(uploaded_file)
            ext_json = extract_put_options_from_image(b64, mtype, api_key_input, model_selected, gemini_proxies)
            st.session_state.raw_options = ext_json.get("put_options", [])
            if not st.session_state.raw_options:
                st.warning("⚠️ 未识别到 Put 期权。")
        except Exception as e:
            st.error(f"解析失败: {e}")

with col2:
    st.subheader("📊 2. 专业风控看板")
    if not st.session_state.raw_options:
        st.info("💡 请在左侧上传截图并点击【开始智能 OCR 分析】")
    else:
        raw_options = st.session_state.raw_options
        
        # 准备数据拉取行情
        underlying_symbols, option_symbols = [], []
        for opt in raw_options:
            u_code = format_market_symbol(opt["underlying_name"])
            o_code = build_longport_option_symbol(u_code, opt["expiration_date"], opt["strike_price"])
            underlying_symbols.append(u_code)
            option_symbols.append(o_code if o_code else "UNKNOWN")

        with st.spinner("正在连接 LongPort 拉取最新价格与希腊字母..."):
            live_u_prices = get_realtime_quotes(underlying_symbols, is_option=False)
            live_o_data = get_realtime_quotes(option_symbols, is_option=True)

        processed_list = []
        tot_notional, tot_premium = 0.0, 0.0

        for idx, opt in enumerate(raw_options):
            qty = abs(int(float(opt["quantity"]))) 
            strike = float(opt["strike_price"])
            cost = float(opt["cost_price"])
            try: dte = max((datetime.strptime(opt["expiration_date"].replace("/", "-"), '%Y-%m-%d') - datetime.now()).days, 0)
            except: dte = 999 

            notional = strike * 100 * qty
            premium = cost * 100 * qty
            tot_notional += notional
            tot_premium += premium

            u_code = underlying_symbols[idx]
            o_code = option_symbols[idx]

            # 取标的现价
            cur_p = live_u_prices.get(u_code, 0.0)
            if cur_p == 0.0:
                try: cur_p = float(re.sub(r'[^\d.]', '', opt["current_price"]))
                except: cur_p = 0.0

            # 取期权现价、IV、Delta
            o_info = live_o_data.get(o_code, {})
            opt_cur_p = o_info.get("price", 0.0)
            iv = o_info.get("iv", 0.0)
            delta = o_info.get("delta", 0.0)
            
            if opt_cur_p == 0.0:
                try: opt_cur_p = float(re.sub(r'[^\d.]', '', opt["current_price"]))
                except: opt_cur_p = cost 

            # 核心风控指标计算
            buffer_pct = ((cur_p - strike) / cur_p * 100) if cur_p > 0 else 0.0
            profit_pct = ((cost - opt_cur_p) / cost * 100) if cost > 0 else 0.0
            ar_pct = (cost / strike) * (365 / max(dte, 1)) * 100 if strike > 0 else 0.0

            # 状态与标签
            status = "🔴 危险" if buffer_pct < 5 or dte < 7 else ("🟡 关注" if buffer_pct < 12 else "🟢 安全")
            tags = []
            if profit_pct >= 50: tags.append("🔥 达标止盈")
            elif profit_pct <= -50: tags.append("⚠️ 深度浮亏")
            if buffer_pct < 0: tags.append("💥 ITM( 破位 )")
            if dte <= 3: tags.append("⏳ 末日临期")
            
            action_tip = get_action_suggestion(profit_pct, buffer_pct, dte, delta)

            processed_list.append({
                "状态": status,
                "标的": u_code,
                "数量": qty,
                "行权价": f"${strike:.2f}",
                "到期日": opt["expiration_date"],
                "DTE": dte,
                "安全垫 (%)": round(buffer_pct, 2),
                "标的现价": f"${cur_p:.2f}",
                "成本价": f"${cost:.2f}",
                "现价": f"${opt_cur_p:.2f}",
                "期权浮盈 (%)": round(profit_pct, 2),
                "年化收益率 (%)": round(ar_pct, 2),
                "Delta": round(delta, 3),
                "IV (%)": round(iv, 2),
                "名义价值": notional,
                "已收权利金": premium,
                "操作建议": action_tip,
                "标记": " | ".join(tags) if tags else "—"
            })

        df = pd.DataFrame(processed_list)

        # ---------------- 顶栏汇总 ----------------
        c1, c2, c3 = st.columns(3)
        c1.metric("🏗️ 总名义暴露 (Notional)", f"${tot_notional:,.2f}")
        c2.metric("💵 总收权利金 (Premium)", f"${tot_premium:,.2f}")
        c3.metric("🎯 整体静态年化 (AR%)", f"{(tot_premium / tot_notional * 365 / max(df['DTE'].mean(), 1) * 100):.2f}%" if tot_notional > 0 else "0.00%")

        st.divider()

        # ---------------- 图表可视化 ----------------
        st.markdown("#### 📈 资金结构可视化")
        vc1, vc2 = st.columns(2)
        with vc1:
            fig_pie = px.pie(df, values='名义价值', names='标的', title='标的资金暴露集中度', hole=0.4)
            fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        with vc2:
            fig_bar = px.bar(df, x='到期日', y='名义价值', color='标的', title='按到期日分布的资金敞口 ( 交割压力 )', text_auto='.2s')
            fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="到期时间", yaxis_title="暴露金额 ($)")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ---------------- 数据大表 ----------------
        st.markdown("#### 🔬 穿透后持仓风控明细")
        df = df.sort_values(by=["状态", "期权浮盈 (%)"])

        def highlight_risk(row):
            if "ITM" in row['标记'] or "危险" in row['状态']: return ['background-color: #ffcccc; color: #900'] * len(row)
            if "浮亏" in row['标记'] or "关注" in row['状态']: return ['background-color: #fff4cc; color: #860'] * len(row)
            if "止盈" in row['标记']: return ['background-color: #ccffcc; color: #060'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df.style.apply(highlight_risk, axis=1).format({
                "名义价值": "${:,.2f}", "已收权利金": "${:,.2f}",
                "安全垫 (%)": "{:.2f}", "期权浮盈 (%)": "{:.2f}",
                "年化收益率 (%)": "{:.2f}", "Delta": "{:.3f}", "IV (%)": "{:.2f}"
            }),
            use_container_width=True, 
            hide_index=True,
            column_config={
                "标记": st.column_config.TextColumn("风险标记 🎯", width="medium"),
                "状态": st.column_config.TextColumn("状态", width="small"),
                "操作建议": st.column_config.TextColumn("💡 AI 操作建议", width="large"),
                "安全垫 (%)": st.column_config.NumberColumn("安全垫 (%)", format="%.2f"),
                "期权浮盈 (%)": st.column_config.NumberColumn("期权浮盈 (%)", format="%.2f"),
                "年化收益率 (%)": st.column_config.NumberColumn("年化收益率 (%)", format="%.2f"),
                "Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                "IV (%)": st.column_config.NumberColumn("IV (%)", format="%.2f")
            }
        )

        st.download_button("📥 一键下载分析数据为 CSV", data=df.to_csv(index=False, encoding="utf-8-sig"), file_name="sell_put_dashboard.csv", mime="text/csv", use_container_width=True)
