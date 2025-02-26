import streamlit as st

st.set_page_config(page_title="Secret Key Checker", layout="centered")

st.title("🔑 Checking Streamlit Secrets")

if "GROQ_API_KEY" in st.secrets:
    st.success("✅ GROQ_API_KEY is available!")
    st.write(f"🔒 API Key: {st.secrets['GROQ_API_KEY'][:5]}********")
else:
    st.error("❌ GROQ_API_KEY is missing! Check your secrets.toml file.")

st.write("👉 Make sure you've added the key in `.streamlit/secrets.toml` or Streamlit Cloud settings.")
