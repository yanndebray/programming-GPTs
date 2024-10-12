import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # Import Plotly Express

# Define possible next tokens
possible_tokens = ["away", "near", "far", "distant", "planet", "star", "universe", "time", "space"]

# Function to generate synthetic probabilities based on temperature and top_p
def generate_synthetic_data(temperature, top_p):
    # Base probabilities (simulated)
    base_probs = np.array([0.2, 0.15, 0.25, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05])
    
    # Apply temperature scaling
    scaled_probs = np.exp(np.log(base_probs + 1e-10) / temperature)
    scaled_probs /= np.sum(scaled_probs)  # Normalize to sum to 1

    # Apply top_p filtering
    sorted_indices = np.argsort(scaled_probs)[::-1]
    cumulative_probs = np.cumsum(scaled_probs[sorted_indices])
    
    # Check if any indices satisfy the top_p condition
    cutoff_indices = np.where(cumulative_probs > top_p)[0]
    
    if len(cutoff_indices) == 0:
        # If no indices found, use all tokens
        cutoff_index = len(scaled_probs) - 1
    else:
        cutoff_index = cutoff_indices[0]

    filtered_probs = np.zeros_like(scaled_probs)
    filtered_probs[sorted_indices[:cutoff_index + 1]] = scaled_probs[sorted_indices[:cutoff_index + 1]]
    filtered_probs /= np.sum(filtered_probs)  # Normalize again

    return possible_tokens, filtered_probs

# Streamlit UI
st.sidebar.title("Temperature and Top P")

st.write("A long time ago in a galaxy far...")

# Sidebar for selecting display option
display_option = st.sidebar.radio("Choose display option:", ("Temperature", "Top P"))

# Set default values for temperature and top_p
if display_option == "Temperature":
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    top_p = 1.0  # Default top_p when temperature is selected
else:
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    temperature = 1.0  # Default temperature when top_p is selected

# Generate synthetic data
tokens, probabilities = generate_synthetic_data(temperature, top_p)

# Create a DataFrame for display
df = pd.DataFrame({'Token': tokens, 'Probability': probabilities})

# Display the results
st.sidebar.write('**Next probable tokens with their probabilities:**')
st.sidebar.write(df)

# Sort DataFrame by 'Probability' column in descending order
df_sorted = df.sort_values(by='Probability', ascending=True)

# Create a Plotly bar chart for probabilities in horizontal orientation
fig = px.bar(df_sorted, x='Probability', y='Token', title=f'Probabilities of Next Tokens (Temperature: {temperature}, Top P: {top_p})',
             labels={'Probability': 'Probability', 'Token': 'Next Token'},
             color='Probability', color_continuous_scale=px.colors.sequential.Viridis)

# Show the Plotly figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
