from groq import Groq
import streamlit as st
from time import perf_counter

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def generate_answer(query, context, model_name):
    start = perf_counter()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer strictly from provided data. "
                    "If not present, say information is not available."
                )
            },
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nData:\n{context}"
            }
        ],
        temperature=0.0
    )

    latency = (perf_counter() - start) * 1000
    return response.choices[0].message.content, latency
