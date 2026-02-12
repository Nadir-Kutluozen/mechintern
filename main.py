import streamlit as st
import components as comp
from models import load_model, get_tokens, get_activations, get_attention_patterns, get_attention_patterns

st.title("ML Experiment Runner")

# --- REFACTORED: Get all settings from sidebar ---
experiment_name, layer_index, number_neurons, sort_order, show_matrix = comp.sideBar(
    "Experiment Name", 
    "Experiment SAE with gpt2-small"
)
# -------------------------------------------------

st.write(f"### Running: {experiment_name}")
st.write(f"Analyzing **Layer {layer_index}** | Top **{number_neurons}** Neurons")

@st.cache_resource
def get_model():
    return load_model()

with st.spinner(f"Loading Model..."):
    model = get_model()
    st.success(f"Model Loaded Successfully!")

text_input = st.text_input("Enter text to analyze:", "Hello World")

if text_input:

    # (Deleted old columns/sliders code from here)

    df_tokens = get_tokens(model, text_input)
    activations = get_activations(model, text_input, layer_index)
    
    comp.show_tokens(df_tokens)
    
    # Pass new config arguments to display function
    comp.show_activations(activations, number_neurons, df_tokens["Token"], sort_order, show_matrix)
    
    # --- Attention Analysis ---
    attention_pattern = get_attention_patterns(model, text_input, layer_index)
    comp.show_attention_pattern(attention_pattern, df_tokens["Token"])
    # --------------------------
