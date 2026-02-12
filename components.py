import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# side bar 
def sideBar(ex_name, default_experiment_name):
    st.sidebar.header("Experiment Settings")
    experiment_name = st.sidebar.text_input(ex_name, default_experiment_name)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Analysis Settings")
    
    # Move Layer Selection here
    layer_index = st.sidebar.slider("Select Layer", 0, 11, 0)
    
    # Move Neuron Selection here
    number_neurons = st.sidebar.slider("Number of Neurons to Show", 1, 3000, 50)
    
    # Move Sort Order here
    sort_order = st.sidebar.selectbox(
        "Sort Neurons By", 
        ["Highest Activation", "Lowest Activation", "Unsorted"],
        index=0
    )
    
    # Move Matrix Toggle here
    show_matrix = st.sidebar.checkbox("Show Raw Matrix (The Math)", value=False)
    
    return experiment_name, layer_index, number_neurons, sort_order, show_matrix

def show_tokens(df_tokens):
    """
    Displays the tokens DataFrame in Streamlit.
    """
    st.subheader("1. Tokenization")
    st.caption("How the model sees your text (broken into tokens).")
    # Convert to string to avoid PyArrow mixed-type error on transpose
    st.dataframe(df_tokens.astype(str).T)


def show_activations(activations, number_neurons, tokens, sort_order, show_matrix):
    """
    Displays the activations in Streamlit.
    """
    
    st.subheader(f"2. Activation Heatmap (Layer Analysis)")
    
    if sort_order == "Unsorted":
            st.info(f"Showing first {number_neurons} neurons in sequential order.")
    else:
            st.caption(f"Showing top {number_neurons} neurons sorted by: {sort_order}")

    
    # Calculate max activation per neuron across all tokens
    max_act_per_neuron = activations.max(axis=0)
    
    # Sort and get indices
    if sort_order == "Highest Activation":
        top_neurons_idx = max_act_per_neuron.sort_values(ascending=False).head(number_neurons).index
    elif sort_order == "Lowest Activation":
        top_neurons_idx = max_act_per_neuron.sort_values(ascending=True).head(number_neurons).index
    else:
        top_neurons_idx = activations.columns[:number_neurons]
    
    # Filter activations to just these neurons
    top_activations = activations[top_neurons_idx]

    # --- Show Raw Matrix (Math) ---
    if show_matrix:
        st.write("### The Raw Matrix (Numbers)")
        st.write("This is the actual *math* behind the heatmap. Each cell is a number representing how 'active' a neuron is.")
        # Create a display version with tokens as index for better readability
        display_df = top_activations.copy()
        display_df.index = tokens # Set index to tokens so rows are labeled
        display_df.columns = [f"Neuron {i}" for i in display_df.columns] # Label columns clearer
        st.dataframe(display_df.style.background_gradient(cmap="viridis", axis=None)) 
    # -----------------------------------
    
    # Calculate global min and max for consistent color scaling
    vmin = activations.min().min()
    vmax = activations.max().max()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Transpose so x-axis is neurons, y-axis is tokens
    # Pass vmin and vmax to lock the color scale
    sns.heatmap(top_activations, cmap="viridis", ax=ax, cbar_kws={'label': 'Activation'}, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Token")
    # Set Y-axis labels to be the actual tokens
    st.pyplot(fig)


def show_attention_pattern(attention_pattern, tokens):
    """
    Displays the attention pattern for a specific head.
    attention_pattern shape: [n_heads, seq_len, seq_len]
    """
    st.markdown("---")
    st.subheader("3. Attention Mechanism (The Connections)")
    
    st.write("""
    **Connection Strength:**
    - **X-Axis (Source):** The word the model is looking *at*.
    - **Y-Axis (Destination):** The word that is 'paying attention'.
    - **Bright Color:** Strong connection (The Y word cares a lot about the X word).
    """)
    
    # Get number of heads from the shape
    n_heads = attention_pattern.shape[0]
    
    # Slider to select head
    head_index = st.slider("Select Attention Head", 0, n_heads - 1, 0)
    
    # Get the matrix for this head
    # Shape: [seq_len, seq_len]
    head_matrix = attention_pattern[head_index]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    # vmin=0, vmax=1 because attention is probability (0 to 1)
    sns.heatmap(head_matrix, cmap="Reds", ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Attention Weight'})
    
    ax.set_xlabel("Source Token (Key)")
    ax.set_ylabel("Destination Token (Query)")
    
    # Set ticks to tokens
    ax.set_xticks(np.arange(len(tokens)) + 0.5)
    ax.set_yticks(np.arange(len(tokens)) + 0.5)
    
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens, rotation=0)
    
    st.pyplot(fig)
