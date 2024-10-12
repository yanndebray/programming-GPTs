import openai
import streamlit as st
import pandas as pd
from math import exp
openai.api_key=st.secrets['OPENAI_API_KEY']
top_logprobs = st.number_input("Top probable tokens", min_value=1, value=3)
@st.cache_data
def get_logprobs(top_logprobs=3):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Complete the following sentence"},
            {"role": "user", "content": "A long time ago in a galaxy far, far away..."},
        ],
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    token = [l.token for l in response.choices[0].logprobs.content]
    logprobs = [c.top_logprobs for c in response.choices[0].logprobs.content]
    return token,logprobs

st.text_input("Prompt", "A long time ago in a galaxy far, far away...",disabled=True)
(token,logprobs) = get_logprobs(top_logprobs)
i = st.slider("Tokens", 0, len(logprobs) - 1)
st.write(''.join(token[:i])) # join list of tokens until i
df = pd.DataFrame([dict(t) for t in logprobs[i]])
# Convert logprobs to probabilities
df['probability'] = df['logprob'].apply(lambda x: exp(x), 3)  # Using e^logprob
st.write('**Next probable tokens:**')
st.write(df[['token', 'probability']])  # Displaying probabilities instead of logprobs
st.bar_chart(df, x="token", y="probability",
             horizontal=True,
             color="token",
             )