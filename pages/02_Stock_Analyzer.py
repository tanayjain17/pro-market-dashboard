# In 02_Stock_Analyzer.py
import streamlit as st
from utils.ml_engine import analyze_indian_stock

st.title("📈 Indian Stock Analyzer")

symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS, HDFCBANK)", "RELIANCE")

if st.button("Analyze"):
    with st.spinner(f"Analyzing {symbol}..."):
        result = analyze_indian_stock(symbol)
        
        if result and 'error' not in result:
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{result['current_price']:.2f}")
            with col2:
                if result['prediction']:
                    st.metric("Stop Loss", f"₹{result['prediction']['sl']:.2f}")
            with col3:
                if result['prediction']:
                    st.metric("Target", f"₹{result['prediction']['tgt']:.2f}")
            
            # Verdict
            if result['prediction']:
                st.markdown(f"### {result['prediction']['verdict']}")
                st.info(f"**Reason:** {result['prediction']['reason']}")
                st.progress(result['prediction']['confidence'])
            
            # Support/Resistance
            st.subheader("📊 Support & Resistance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Support", f"₹{result['support_resistance']['support']}")
            with col2:
                st.metric("Resistance", f"₹{result['support_resistance']['resistance']}")
            
            # News
            if result['news']:
                st.subheader("📰 Latest News")
                for news in result['news']:
                    st.write(f"• {news.get('title', '')}")
        else:
            st.error(f"Could not analyze {symbol}")
