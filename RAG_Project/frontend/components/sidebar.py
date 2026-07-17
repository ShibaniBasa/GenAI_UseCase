
import streamlit as st


def show_sidebar():

    with st.sidebar:

        st.title("Enterprise RAG")

        st.markdown("---")

        st.subheader("Knowledge Base")

        st.success("Vector Database Ready")

        st.write("**Document Loaded:**")
        st.info("Bank_Document.pdf")

        st.markdown("---")

        st.subheader("Chat")

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        st.caption("Ask questions about the uploaded PDF.")