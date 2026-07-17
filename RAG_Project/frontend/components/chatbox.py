import streamlit as st
import requests

# Backend API
API_URL = "http://127.0.0.1:8000/chat"


def show_chatbox():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    question = st.chat_input("Ask a question about the document...")

    if question:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"question": question},
                        timeout=120
                    )
                    response.raise_for_status()

                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {}

                    answer = payload.get("answer") or payload.get("response")
                    if not answer:
                        answer = response.text.strip() or "No answer returned by the backend."

                except Exception as e:
                    answer = f"Connection Error: {str(e)}"

                st.markdown(answer)

        # Store assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        st.rerun()