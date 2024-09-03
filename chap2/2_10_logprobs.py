import openai
import streamlit as st
import pandas as pd
openai.api_key=st.secrets['OPENAI_API_KEY']
top_logprobs = 3
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
    # zero = response.choices[0].logprobs.content[0].top_logprobs
    # st.sidebar.write(zero)
    # df0 = pd.DataFrame(response.choices[0].logprobs.content[:3])
    # st.sidebar.write(df0)
    token = [l.token for l in response.choices[0].logprobs.content]
    logprobs = [c.top_logprobs for c in response.choices[0].logprobs.content]
    return token,logprobs

st.text_input("Prompt", "A long time ago in a galaxy far, far away...",disabled=True)
(token,logprobs) = get_logprobs()
i = st.slider("Tokens", 0, len(logprobs) - 1)
st.write(''.join(token[:i])) # join list of tokens until i
df = pd.DataFrame([dict(t) for t in logprobs[i]])
st.write('**Next probable tokens:**')
st.write(df)
st.bar_chart(df, x="token", y="logprob",
             horizontal=True,
             color="token",
             )