# test_app.py - Minimal app to test
import streamlit as st

st.title("Test App")
st.write("If you see this, Streamlit works!")

# Test imports
try:
    from utils.ml_engine import analyze_indian_stock
    st.success("✅ ML Engine imports OK")
except Exception as e:
    st.error(f"❌ ML Engine import failed: {e}")

try:
    from utils.data_engine import robust_yf_download
    st.success("✅ Data Engine imports OK")
except Exception as e:
    st.error(f"❌ Data Engine import failed: {e}")
