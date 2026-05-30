import os
import re
import base64
import json
import requests
import pandas as pd
from datetime import datetime
import streamlit as st
from longport.openapi import Config, QuoteContext

# ==========================================
# 1. 页面基本配置与美化
# ==========================================
st.set_page_config(
    page_title="Sell Put 智能识别与风险监控舱",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Sell Put 智能识别与实时风险监控舱")
st.markdown("通过上传期权持仓截图，结合 **Gemini 3.5 多模态 OCR** 与 **LongPort OpenAPI 实时行情**，全自动进行穿透式名义价值与安全边际监控。")

# ==========================================
# 2. 侧边栏配置区 ( 读取云端 Secrets 配置 )
# ==========================================
st.sidebar.header("⚙️ 1. Gemini API 配置")

# 从 Streamlit Secrets 中读取默认值
default_gemini_key = st.secrets.get("GEMINI_API_KEY", "")

# API 密钥输入
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value=default_gemini_key,
    type="password",
    help="请输入您的 Gemini API Key"
)

# 模型选择
model_options = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
model_selected = st.sidebar.selectbox(
    "选择分析模型",
    options=model_options,
    index=0,
    help="gemini-3.5-flash 拥有极佳的多模态解析与速率限制表现"
)

# 网络代理配置（云端部署默认关闭代理）
use_proxy = st.sidebar.checkbox("启用网络代理 (Gemini)", value=False)
proxy_url = st.sidebar.text_input(
    "代理服务器地址",
    value="http://127.0.0.1:1601",
    disabled=not use_proxy
)

# 构造 Requests 代理参数
gemini_proxies = None
if use_proxy and proxy_url:
    gemini_proxies = {
        "http": proxy_url,
        "https": proxy_url
    }

st.sidebar.divider()
st.sidebar.header("🔌 2. LongPort API 实时行情配置")

# 从 Streamlit Secrets 中读取长桥默认值
default_lp_app_key = st.secrets.get("LONGPORT_APP_KEY", "")
default_lp_app_secret = st.secrets.get("LONGPORT_APP_SECRET", "")
default_lp_access_token = st.secrets.get("LONGPORT_ACCESS_TOKEN", "")

lp_app_key = st.sidebar.text_input(
    "LongPort APP KEY",
    value=default_lp_app_key,
    type="password"
)
lp_app_secret = st.sidebar.text_input(
    "LongPort APP SECRET",
    value=default_lp_app_secret,
    type="password"
)
lp_access_token = st.sidebar.text_input(
    "LongPort ACCESS TOKEN",
    value=default_lp_access_token,
    type="password"
)

# ==========================================
# 3. LongPort 资源连接池管理
# ==========================================
@st.cache_resource
def get_longport_config(app_key, app_secret, access_token):
    if not (app_key and app_secret and access_token):
        return None
    try:
        return Config(app_key=app_key, app_secret=app_secret, access_token=access_token)
    except Exception as e:
        st.sidebar.error(f"LongPort Config 初始化失败: {e}")
        return None

@st.cache_resource
def get_quote_context(app_key, app_secret, access_token):
    cfg = get_longport_config(app_key, app_secret, access_token)
    if not cfg:
        return None
    try:
        return QuoteContext(cfg)
    except Exception as e:
        st.sidebar.error(f"LongPort 行情连接失败: {e}")
        return None

# 初始化行情上下文
quote_ctx = get_quote_context(lp_app_key, lp_app_secret, lp_access_token)

if quote_ctx:
    st.sidebar.success("✅ LongPort 实时行情连接已就绪")
else:
    st.sidebar.warning("⚠️ 行情未连接。将默认使用 Gemini OCR 提取的静态价格。")

# ==========================================
# 4. 辅助函数：符号转换与数据规整
# ==========================================
def format_market_symbol(symbol):
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    if re.match(r"^\d+$", symbol):
        return f"{symbol.zfill(5)}.HK"
    else:
        return f"{symbol}.US"

def build_longport_option_symbol(underlying, expiry, strike, option_type="P"):
    try:
        underlying = underlying.split(".")[0].upper()
        clean_expiry = expiry.replace("-", "").replace("/", "")
        if len(clean_expiry) == 8:  
            clean_expiry = clean_expiry[2:]
        elif len(clean_expiry) != 6:
            return None

        strike_val = float(strike)
        strike_str = f"{int(strike_val * 1000):08d}"
        suffix = "US" if not re.match(r"^\d+$", underlying) else "HK"
        return f"{underlying}{clean_expiry}{option_type}{strike_str}.{suffix}"
    except Exception:
        return None

def encode_uploaded_file(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    encoded_string = base64.b64encode(bytes_data).decode('utf-8')
    mime_type = uploaded_file.type
    return encoded_string, mime_type

# ==========================================
# 5. Gemini 结构化 OCR 识别逻辑
# ==========================================
def extract_put_options_from_image(base64_data, mime_type, api_key, model_name, proxies):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "put_options": {
                "type": "ARRAY",
                "description": "从图片中提取出的所有 Put 期权（认沽期权）数据列表",
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

    prompt = """
    请仔细辨认这张持仓截图或交易表格。
    过滤并寻找图片中所有的 Put 期权（中文可能显示为“认沽”、“沽”、或代码中带有“P”标识，如 AAPL260618P00150000）。
    准确提取出这些 Put 期权对应的：
    1. 标的代码或名称
    2. 行权价
    3. 到期时间：尽量整理为 YYYY-MM-DD 格式
    4. 持仓数量：如果是卖出开仓（Sell Put），数量一般表示为负数（例如 -1，-5）。
    5. 期权现价
    6. 成本均价

    如果识别到多个 Put 期权，请全部列出。不要把 Call（认购期权）或者普通股票正股数据混入。
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": mime_type, "data": base64_data}}]}],
        "systemInstruction": {"parts": [{"text": "你是一个高精度的金融期权 OCR 数据提取助手。你能够精准识别持仓图像中的各项期权交易指标，并严格按指定的 JSON 结构输出。确保不遗漏负号等核心细节。"}]},
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": response_schema}
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=30)
    
    if response.status_code == 200:
        result_json = response.json()
        text_content = result_json['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_content)
    else:
        raise RuntimeError(f"请求失败，状态码: {response.status_code}，错误信息: {response.text}")

# ==========================================
# 6. 长桥实时价格获取与回贴逻辑
# ==========================================
def get_realtime_quotes(symbols, is_option=False):
    if not quote_ctx or not symbols:
        return {}
    try:
        clean_symbols = list(set([s for s in symbols if s and s != "UNKNOWN"]))
        if not clean_symbols:
            return {}
        quotes = quote_ctx.option_quote(clean_symbols) if is_option else quote_ctx.quote(clean_symbols)
        return {q.symbol: float(q.last_done) for q in quotes}
    except Exception as e:
        st.warning(f"获取实时价格失败: {e}，将采用 OCR 识别价格替代。")
        return {}

# ==========================================
# 7. 主页面交互与展示
# ==========================================
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📸 1. 上传期权截图")
    uploaded_file = st.file_uploader("支持 PNG, JPG, JPEG 等格式的持仓/交易截图", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="已上传的期权截图", use_container_width=True)

with col2:
    st.subheader("📊 2. 实时穿透分析结果")
    if uploaded_file is None:
        st.info("💡 请先在左侧上传期权截图，系统将实时计算您的 Sell Put 暴露及风险。")
    else:
        if not api_key_input:
            st.error("❌ 请先在侧边栏配置您的 Gemini API Key。")
        else:
            if st.button("🚀 开始 OCR 识别并拉取实时行情", type="primary", use_container_width=True):
                with st.spinner("第一步：正在调用 Gemini 3.5 提取结构化持仓指标..."):
                    try:
                        base64_data, mime_type = encode_uploaded_file(uploaded_file)
                        extracted_json = extract_put_options_from_image(
                            base64_data=base64_data, mime_type=mime_type, api_key=api_key_input,
                            model_name=model_selected, proxies=gemini_proxies
                        )
                        raw_options = extracted_json.get("put_options", [])
                    except Exception as e:
                        st.error(f"Gemini OCR 提取失败: {e}")
                        raw_options = []

                    if not raw_options:
                        st.warning("⚠️ 未能在此图片中成功匹配或识别到任何 Put（认沽）期权持仓。")
                    else:
                        st.success(f"🎉 成功识别出 {len(raw_options)} 个期权持仓！正在进行长桥行情穿透...")
                        
                        processed_list = []
                        underlying_symbols = []
                        option_symbols = []

                        for opt in raw_options:
                            raw_underlying = opt["underlying_name"]
                            formatted_underlying = format_market_symbol(raw_underlying)
                            underlying_symbols.append(formatted_underlying)
                            opt_symbol = build_longport_option_symbol(formatted_underlying, opt["expiration_date"], opt["strike_price"])
                            option_symbols.append(opt_symbol if opt_symbol else "UNKNOWN")

                        live_underlying_prices = {}
                        live_option_prices = {}

                        if quote_ctx:
                            with st.spinner("第二步：正在向长桥拉取标的及期权最新实时价..."):
                                live_underlying_prices = get_realtime_quotes(underlying_symbols, is_option=False)
                                live_option_prices = get_realtime_quotes(option_symbols, is_option=True)

                        total_notional_exposure = 0.0
                        total_premium_received = 0.0

                        for idx, opt in enumerate(raw_options):
                            qty = abs(int(float(opt["quantity"]))) 
                            strike = float(opt["strike_price"])
                            cost = float(opt["cost_price"])

                            try:
                                dte = max((datetime.strptime(opt["expiration_date"].replace("/", "-"), '%Y-%m-%d') - datetime.now()).days, 0)
                            except Exception:
                                dte = 999 

                            notional = strike * 100 * qty
                            total_notional_exposure += notional
                            premium = cost * 100 * qty
                            total_premium_received += premium

                            underlying_code = underlying_symbols[idx]
                            option_code = option_symbols[idx]

                            cur_p = live_underlying_prices.get(underlying_code, 0.0)
                            if cur_p == 0.0:
                                try:
                                    cur_p = float(re.sub(r'[^\d.]', '', opt["current_price"]))
                                except Exception:
                                    cur_p = 0.0

                            opt_cur_p = live_option_prices.get(option_code, 0.0)
                            if opt_cur_p == 0.0:
                                try:
                                    opt_cur_p = float(re.sub(r'[^\d.]', '', opt["current_price"]))
                                except Exception:
                                    opt_cur_p = cost 

                            buffer_pct = ((cur_p - strike) / cur_p * 100) if cur_p > 0 else 0.0
                            profit_pct = ((cost - opt_cur_p) / cost * 100) if cost > 0 else 0.0

                            if buffer_pct < 5 or dte < 7:
                                status = "🔴 危险 ( 高危/末日 )"
                            elif buffer_pct < 12:
                                status = "🟡 关注 ( 警戒 )"
                            else:
                                status = "🟢 安全"

                            # 丰富【标记】列的业务逻辑（支持多标签组合）
                            tags = []
                            if profit_pct >= 50:
                                tags.append("🔥 达标止盈 (>50%)")
                            elif profit_pct <= -50:
                                tags.append("⚠️ 深度浮亏")
                                
                            if buffer_pct < 0:
                                tags.append("💥 跌破行权价 (ITM)")
                            
                            if dte <= 3:
                                tags.append("⏳ 末日临期")

                            target_tag = " | ".join(tags) if tags else "—"

                            processed_list.append({
                                "状态": status,
                                "标的": underlying_code,
                                "数量 ( 张 )": qty,
                                "行权价": f"${strike:.2f}",
                                "到期日": opt["expiration_date"],
                                "剩余天数 (DTE)": dte,
                                "标的现价": f"${cur_p:.2f}",
                                "期权成本价": f"${cost:.2f}",
                                "最新期权价": f"${opt_cur_p:.2f}",
                                "名义价值 (USD)": notional,
                                "已收权利金": premium,
                                "安全垫 (%)": round(buffer_pct, 2),
                                "期权浮盈 (%)": round(profit_pct, 2),
                                "标记": target_tag
                            })

                        df = pd.DataFrame(processed_list)
                        st.markdown("### 💰 资金暴露一览")
                        c1, c2 = st.columns(2)
                        c1.metric("🏗️ 识别持仓名义总暴露", f"${total_notional_exposure:,.2f}", help="如果期权被完全指派买入正股，需要付出的现金总额")
                        c2.metric("💵 识别持仓已收总权利金", f"${total_premium_received:,.2f}", help="未平仓前已斩获的安全垫权利金")

                        st.markdown("### 📊 穿透后持仓风控明细阵列")
                        df = df.sort_values(by=["期权浮盈 (%)", "安全垫 (%)"])

                        # 优化高亮逻辑，覆盖新标签
                        def highlight_risk(row):
                            if "ITM" in row['标记'] or "危险" in row['状态']: 
                                return ['background-color: #ffcccc; color: #900'] * len(row)
                            if "浮亏" in row['标记'] or "关注" in row['状态']: 
                                return ['background-color: #fff4cc; color: #860'] * len(row)
                            if "止盈" in row['标记']: 
                                return ['background-color: #ccffcc; color: #060'] * len(row)
                            return [''] * len(row)

                        # 使用 column_config 和 format 强制限制小数位数和列宽
                        st.dataframe(
                            df.style.apply(highlight_risk, axis=1).format({
                                "名义价值 (USD)": "${:,.2f}", 
                                "已收权利金": "${:,.2f}",
                                "安全垫 (%)": "{:.2f}",      # 强制格式化为 2 位小数
                                "期权浮盈 (%)": "{:.2f}"     # 强制格式化为 2 位小数
                            }),
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "标记": st.column_config.TextColumn("标记 🎯", width="large"),
                                "状态": st.column_config.TextColumn("状态", width="medium"),
                                "安全垫 (%)": st.column_config.NumberColumn("安全垫 (%)", format="%.2f"),
                                "期权浮盈 (%)": st.column_config.NumberColumn("期权浮盈 (%)", format="%.2f")
                            }
                        )

                        csv_data = df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button("📥 一键下载分析数据为 CSV", data=csv_data, file_name="extracted_options.csv", mime="text/csv", use_container_width=True)
