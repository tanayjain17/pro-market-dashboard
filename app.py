# app.py
import nltk
import ssl
import streamlit as st
import pandas as pd
import feedparser

# ==================== NLTK SETUP (MUST BE FIRST) ====================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==================== IMPORTS (AFTER NLTK) ====================
from utils.data_engine import robust_yf_download, get_fundamentals, get_news_sentiment
from utils.ml_engine import calculate_features, run_smart_prediction, analyze_indian_stock
from utils.chart_engine import plot_candlestick, plot_macd
from utils.analytics_engine import init_analytics_db, log_prediction, calculate_win_rate

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Pro Market Dashboard",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# ==================== CSS STYLING ====================
st.markdown("""
<style>
    .stApp { background-color: #0f1115; font-family: 'Inter', sans-serif; }
    .fun-card {
        background: rgba(30, 34, 45, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 15px;
        cursor: pointer;
        transition: 0.3s;
    }
    .fun-card:hover { transform: translateY(-3px); border-color: #00d09c; }
    .metric-container {
        background: #1e2330;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-label { font-size: 10px; color: #aaa; text-transform: uppercase; }
    .metric-value { font-size: 14px; font-weight: bold; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE DATABASE ====================
init_analytics_db()

# ==================== SESSION STATE ====================
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = "RELIANCE.NS"

# ==================== MAIN APP ====================
st.title("🚀 Pro Market Dashboard")

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📈 Stock Analyzer",
    "📰 Market News",
    "⭐ AI Picks"
])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.header("Market Overview")
    
    INDICES = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "BANK NIFTY": "^NSEBANK"
    }
    
    cols = st.columns(3)
    for (name, sym), col in zip(INDICES.items(), cols):
        with col:
            df = robust_yf_download(sym, "5d")
            if df is not None and not df.empty:
                curr = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2] if len(df) > 1 else curr
                chg = curr - prev
                pct = (chg / prev) * 100 if prev != 0 else 0
                clr = "#00d09c" if chg >= 0 else "#ff4b4b"
                
                st.markdown(f"""
                <div class="fun-card" style="border-top:3px solid {clr}">
                    <div style="color:#aaa; font-size:12px;">{name}</div>
                    <div style="font-size:22px; font-weight:bold;">₹{curr:,.2f}</div>
                    <div style="color:{clr}; font-weight:bold;">{chg:+.2f} ({pct:+.2f}%)</div>
                </div>
                """, unsafe_allow_html=True)

# ==================== TAB 2: STOCK ANALYZER ====================
with tab2:
    st.header("Stock Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        market = st.selectbox("Market", ["NSE", "BSE"])
    with col2:
        ticker_input = st.text_input("Ticker Symbol", "RELIANCE")
    
    full_ticker = f"{ticker_input.upper()}.NS" if market == "NSE" else f"{ticker_input.upper()}.BO"
    
    if st.button("🔍 Analyze", key="analyze"):
        with st.spinner(f"Analyzing {full_ticker}..."):
            try:
                # Try using our new Indian stock analyzer first
                result = analyze_indian_stock(ticker_input.upper(), market)
                
                if result and 'error' not in result and result['prediction']:
                    prediction = result['prediction']
                    
                    # Verdict Card
                    st.markdown(f"""
                    <div class="fun-card" style="border-left: 8px solid {prediction['color']}; margin-bottom: 20px;">
                        <h1 style="color:{prediction['color']}; margin:0;">{prediction['verdict']}</h1>
                        <p style="color:#aaa;">Confidence: {prediction['confidence']*100:.0f}% | Source: {result.get('data_source', 'Free API')}</p>
                        <hr style="border-color:#333;">
                        <p style="color:#ddd;"><i>{prediction['reason']}</i></p>
                        <div style="display:flex; justify-content:space-around; margin-top:10px;">
                            <div><span style="color:#aaa;">Current</span><h3>₹{prediction['curr']:.2f}</h3></div>
                            <div><span style="color:#aaa;">Stop Loss</span><h3 style="color:#ff4b4b;">₹{prediction['sl']:.2f}</h3></div>
                            <div><span style="color:#aaa;">Target</span><h3 style="color:#00d09c;">₹{prediction['tgt']:.2f}</h3></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Support & Resistance
                    if result.get('support_resistance'):
                        sr = result['support_resistance']
                        col1, col2 = st.columns(2)
                        col1.metric("Support Level", f"₹{sr['support']}")
                        col2.metric("Resistance Level", f"₹{sr['resistance']}")
                    
                    # Log prediction if BUY
                    if "BUY" in prediction['verdict']:
                        pred_id = log_prediction(
                            ticker=full_ticker,
                            verdict=prediction['verdict'],
                            predicted_price=prediction['curr'],
                            sl=prediction['sl'],
                            tgt=prediction['tgt'],
                            confidence=prediction['confidence'],
                            entry_price=prediction['curr']
                        )
                        
                        # Show win rate
                        wr = calculate_win_rate(days=30)
                        st.info(f"📊 30-day Win Rate: **{wr['win_rate']:.1f}%** ({wr['wins']}/{wr['total_trades']} trades)")
                    
                    # News section
                    if result.get('news'):
                        with st.expander("📰 Latest News"):
                            for news in result['news'][:3]:
                                st.markdown(f"• {news.get('title', '')}")
                
                else:
                    # Fallback to yfinance
                    st.warning("Using yfinance as fallback...")
                    df = robust_yf_download(full_ticker, "2y")
                    
                    if df is not None:
                        df = calculate_features(df)
                        if df is not None:
                            prediction = run_smart_prediction(full_ticker, df)
                            if prediction:
                                st.markdown(f"""
                                <div class="fun-card" style="border-left: 8px solid {prediction['color']};">
                                    <h1 style="color:{prediction['color']};">{prediction['verdict']}</h1>
                                    <p>{prediction['reason']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Charts
                                st.subheader("📉 Price Action")
                                st.plotly_chart(plot_candlestick(df, full_ticker), use_container_width=True)
                                
                                st.subheader("📊 MACD Indicator")
                                st.plotly_chart(plot_macd(df), use_container_width=True)
                            else:
                                st.error("Could not generate prediction.")
                        else:
                            st.error("Error calculating features.")
                    else:
                        st.error(f"Could not fetch data for {full_ticker}")
                        
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ==================== TAB 3: MARKET NEWS ====================
with tab3:
    st.header("📰 Latest Market News")
    
    try:
        feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
        for entry in feed.entries[:10]:
            with st.container():
                st.markdown(f"""
                <div style="background:#161920; padding:12px; border-radius:10px; margin-bottom:10px; border-left:3px solid #4c8bf5;">
                    <a href='{entry.link}' target='_blank' style='color:#00d09c;text-decoration:none;'><b>{entry.title}</b></a>
                    <p style='color:#aaa; font-size:12px; margin-top:5px;'>{entry.published}</p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not fetch news: {e}")

# ==================== TAB 4: AI PICKS ====================
with tab4:
    st.header("⭐ AI Stock Picks")
    
    SCANNER_POOL = [
        "RELIANCE", "HDFCBANK", "INFY", "TCS",
        "ITC", "SBIN", "TATAMOTORS", "ZOMATO"
    ]
    
    if st.button("🔍 Scan All Stocks"):
        progress_bar = st.progress(0)
        results = []
        
        for i, ticker in enumerate(SCANNER_POOL):
            progress_bar.progress((i + 1) / len(SCANNER_POOL))
            
            result = analyze_indian_stock(ticker, "NSE")
            if result and 'error' not in result and result['prediction']:
                pred = result['prediction']
                if "BUY" in pred['verdict']:
                    results.append((ticker, pred['verdict'], pred['curr'], pred['color']))
        
        progress_bar.empty()
        
        if results:
            st.subheader(f"✅ Found {len(results)} Buy Signals")
            for ticker, verdict, curr, color in results:
                st.markdown(f"""
                <div class='fun-card' style='border-left:5px solid {color}'>
                    <b>{ticker}</b>: {verdict} @ ₹{curr:.2f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No buy signals found today.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:11px; margin-top:30px; border-top:1px solid #333; padding-top:10px;">
    <b>⚠️ Disclaimer:</b> This AI dashboard is for educational purposes only. 
    It is NOT financial advice. Always consult a qualified financial advisor before trading.
    Markets are highly volatile and predictions can fail.
</div>
""", unsafe_allow_html=True)
