from groq import Groq
from time import perf_counter

def generate_answer(query, context, model_name, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
Question:
{query}

Data:
{context}
"""

    start = perf_counter()

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer strictly from the provided data. "
                    "If the answer is not present, say it is not available."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_completion_tokens=1024
    )

    latency_ms = (perf_counter() - start) * 1000

    return completion.choices[0].message.content, latency_ms
