import openai
import streamlit as st
import pandas as pd
openai.api_key=st.secrets['OPENAI_API_KEY']
st.sidebar.subheader('💸 Cost management')
df = pd.read_csv('chap2/prices.csv',index_col=0)
st.sidebar.table(df)

model = st.selectbox('Select a model',df.index)
col1,col2 = st.columns(2)
with col1:
    prompt = st.text_input('Enter your message','Hello')
with col2:
    max_tokens = st.number_input('Max tokens',
        min_value=0,
        max_value=128_000,
        value=10
    )

if prompt:
    response = openai.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{'role': 'user','content': prompt}])
    st.write(response.choices[0].message.content)
    st.write(response.usage)
    # st.sidebar.table(pd.DataFrame(response.usage))

    gpt_4o_mini_input_cost = df[df.index == model]['input'][0]/1_000_000
    gpt_4o_mini_output_cost = df[df.index == model]['output'][0]/1000_000

    st.write('Total cost (in $):',gpt_4o_mini_input_cost + gpt_4o_mini_output_cost)
    # st.write(response)