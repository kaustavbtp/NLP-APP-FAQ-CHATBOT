import streamlit as st
import requests

st.title("FAQ Chatbot")

backend_url = "https://my-faq-app-rested-camel-vg.cfapps.us10-001.hana.ondemand.com"  # Change this to your deployed FastAPI backend URL

def get_answer(question):
    payload = {"nlp": {"source": question}}
    response = requests.post(f"{backend_url}/kba", json=payload)
    if response.status_code == 200:
        return response.json().get("replies")[0].get("content")
    else:
        return "Sorry, something went wrong. Please try again."

question = st.text_input("Ask a question:")
if st.button("Submit"):
    if question:
        answer = get_answer(question)
        st.write("Answer:", answer)
    else:
        st.write("Please enter a question.")