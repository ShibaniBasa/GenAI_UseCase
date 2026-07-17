import streamlit as st

from components.header import show_header
from components.sidebar import show_sidebar
from components.chatbox import show_chatbox
from components.footer import show_footer

def show_chat_page():

    show_sidebar()

    show_header()

    show_chatbox()

    show_footer()