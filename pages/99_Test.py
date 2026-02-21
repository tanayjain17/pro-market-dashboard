# pages/99_Test.py
import streamlit as st
import requests
import sys

st.title("🔧 Debug Page")

st.subheader("System Info")
st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")

# Test API connection
st.subheader("Test Free API")
try:
    response = requests.get(
        "https://military-jobye-haiqstudios-14f59639.koyeb.app/stock",
        params={"symbol": "RELIANCE.NS", "res": "num"},
        timeout=5
    )
    if response.status_code == 200:
        st.success("✅ API is working!")
        st.json(response.json())
    else:
        st.error(f"❌ API returned {response.status_code}")
except Exception as e:
    st.error(f"❌ API error: {e}")

# Test imports
st.subheader("Test Imports")
try:
    from utils.ml_engine import analyze_indian_stock
    st.success("✅ ml_engine imported successfully")
except Exception as e:
    st.error(f"❌ Import failed: {e}")
