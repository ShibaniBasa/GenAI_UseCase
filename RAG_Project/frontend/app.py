##Frontend app.py
import streamlit as st
from pages.Chat import show_chat_page

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Load Custom CSS
# --------------------------------------------------

def load_css():
    try:
        with open("styles/style.css") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        pass

load_css()

# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Open Chat Page
# --------------------------------------------------

show_chat_page()