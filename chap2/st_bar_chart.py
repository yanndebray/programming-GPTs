import streamlit as st
import pandas as pd
import numpy as np
from vega_datasets import data
chart_data = data.barley()
# chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
# chart_data = pd.DataFrame(
#     {"col1": list(range(20)) * 3, 
#      "col2": np.random.randn(60),
#      "col3": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
#      }
# )

if st.toggle('dataframe', True):
    st.dataframe(chart_data)
else:
    st.table(chart_data)

# st.bar_chart(chart_data)
# st.bar_chart(chart_data, x="col1", y="col2", color="col3")
st.bar_chart(chart_data, x="year", y="yield", color="site", 
            #  horizontal=True, 
             stack=False)
