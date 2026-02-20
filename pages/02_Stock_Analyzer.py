# 02_Stock_Analyzer.py
import streamlit as st
from utils.ml_engine import analyze_indian_stock

# Page config
st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Indian Stock Analyzer")
st.markdown("Analyze Indian stocks with technical indicators and AI-powered predictions")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input(
        "Enter Stock Symbol", 
        value="RELIANCE",
        help="Examples: RELIANCE, TCS, HDFCBANK, INFY, ITC"
    ).upper()
with col2:
    exchange = st.selectbox(
        "Exchange",
        options=["NSE", "BSE"],
        index=0,
        help="NSE for National Stock Exchange, BSE for Bombay Stock Exchange"
    )

# Analyze button
if st.button("🔍 Analyze Stock", type="primary", use_container_width=True):
    with st.spinner(f"🔄 Analyzing {symbol}... This may take a few seconds"):
        
        # Call the function from ml_engine.py
        result = analyze_indian_stock(symbol, exchange)
        
        # Check if analysis was successful
        if result and 'error' not in result:
            
            # Success message
            st.success(f"✅ Analysis complete for {symbol}")
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"₹{result['current_price']:.2f}",
                    help="Latest trading price"
                )
            
            if result['prediction']:
                with col2:
                    st.metric(
                        "Stop Loss", 
                        f"₹{result['prediction']['sl']:.2f}",
                        delta="-2%",
                        delta_color="inverse",
                        help="Recommended stop loss level"
                    )
                with col3:
                    st.metric(
                        "Target", 
                        f"₹{result['prediction']['tgt']:.2f}",
                        delta="+3%",
                        help="Recommended target price"
                    )
                with col4:
                    confidence_pct = result['prediction']['confidence'] * 100
                    st.metric(
                        "Confidence", 
                        f"{confidence_pct:.0f}%",
                        help="Prediction confidence score"
                    )
            
            # Verdict section
            st.divider()
            
            if result['prediction']:
                verdict = result['prediction']['verdict']
                reason = result['prediction']['reason']
                color = result['prediction']['color']
                
                # Display verdict with color
                st.markdown(
                    f"<h2 style='color: {color}; text-align: center;'>{verdict}</h2>",
                    unsafe_allow_html=True
                )
                st.markdown(f"<h4 style='text-align: center;'>{reason}</h4>", unsafe_allow_html=True)
                
                # Confidence bar
                st.progress(result['prediction']['confidence'], text="Confidence Level")
                
                # Technical indicators
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RSI (14)", f"{result['prediction']['rsi']:.1f}")
                with col2:
                    st.metric("MACD Histogram", f"{result['prediction']['macd']:.3f}")
            
            # Support & Resistance section
            st.divider()
            st.subheader("📊 Support & Resistance Levels")
            
            col1, col2 = st.columns(2)
            with col1:
                support = result['support_resistance']['support']
                st.metric(
                    "Support Level",
                    f"₹{support:.2f}",
                    help="Price level where stock tends to find buying interest"
                )
            with col2:
                resistance = result['support_resistance']['resistance']
                st.metric(
                    "Resistance Level",
                    f"₹{resistance:.2f}",
                    help="Price level where stock tends to face selling pressure"
                )
            
            # News section
            if result['news']:
                st.divider()
                st.subheader(f"📰 Latest News & Sentiment")
                
                # Show sentiment
                sentiment = result['news_sentiment']
                sentiment_color = {
                    "POSITIVE": "🟢",
                    "NEGATIVE": "🔴",
                    "NEUTRAL": "⚪"
                }.get(sentiment, "⚪")
                
                st.info(f"**Market Sentiment:** {sentiment_color} {sentiment}")
                
                # Show news headlines
                for idx, news in enumerate(result['news'][:5], 1):
                    title = news.get('title', 'No title available')
                    link = news.get('link', '#')
                    published = news.get('published', 'Recent')
                    
                    with st.container():
                        st.markdown(f"**{idx}. {title}**")
                        st.caption(f"🕒 {published}")
                        if link != '#':
                            st.markdown(f"[Read more]({link})")
                        st.divider()
            
            # Timestamp
            st.caption(f"Last updated: {result['timestamp']}")
            
        else:
            # Error message
            st.error(f"❌ Could not analyze {symbol}. Please check:")
            st.info("• Stock symbol is correct (e.g., RELIANCE, TCS, HDFCBANK)")
            st.info("• Internet connection is working")
            st.info("• Try again in a few seconds")

# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Enter Symbol**: Type Indian stock symbol (e.g., RELIANCE, TCS)
    2. **Select Exchange**: Choose NSE or BSE
    3. **Click Analyze**: Get instant analysis
    
    ### Popular Symbols
    - RELIANCE
    - TCS
    - HDFCBANK
    - INFY
    - ITC
    - WIPRO
    - SBIN
    - BHARTIARTL
    """)
    
    st.divider()
    
    st.header("📈 Indicators Explained")
    st.markdown("""
    - **RSI**: Relative Strength Index (oversold < 30, overbought > 70)
    - **MACD**: Moving Average Convergence Divergence
    - **Support**: Price floor where buying occurs
    - **Resistance**: Price ceiling where selling occurs
    """)
    
    st.divider()
    
    st.caption("🚀 Powered by Free Indian Stock API + Machine Learning")
