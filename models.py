from transformer_lens import HookedTransformer
import pandas as pd

def load_model():
    """
    Load the models as option later to test different models"""
    model = HookedTransformer.from_pretrained("gpt2-small")
    return model

def get_tokens(model, text):
    """
    Tokenizes the text and returns a DataFrame with tokens and their IDs.
    """
    tokens = model.to_tokens(text)
    str_tokens = model.to_str_tokens(text)
    # turn into String
    str_tokens = [str(t) for t in str_tokens]
    
    # Convert tensor to numpy for DataFrame
    tokens_np = tokens[0].cpu().numpy()
    
    df_tokens = pd.DataFrame({
        "Token": str_tokens,
        "ID": tokens_np
    })
    return df_tokens

def get_activations(model, text, layer_index=0):
    """
    Returns the activations of the model for the given text and layer.
    """
    # run_with_cache returns (logits, cache)
    _, cache = model.run_with_cache(text)
    
    # Common hook name: "blocks.{layer_index}.mlp.hook_post"
    layer_name = f"blocks.{layer_index}.mlp.hook_post" 
    
    # Get the tensor: shape [batch, pos, d_mlp]
    activations = cache[layer_name]
    
    # Squeeze batch dimension if it's 1
    activations = activations.squeeze(0) # shape [pos, d_mlp]
    
    # Convert to numpy for display
    # We need to detach from the graph and move to cpu
    return pd.DataFrame(activations.detach().cpu().numpy())


def get_attention_patterns(model, text, layer_index=0):
    """
    Returns the attention patterns (post-softmax) for the given layer.
    Shape: [n_heads, seq_len, seq_len]
    """
    # run_with_cache returns (logits, cache)
    _, cache = model.run_with_cache(text)
    
    # Hook name: "blocks.{layer_index}.attn.hook_pattern"
    # This gives the attention probability matrix (post softmax)
    layer_name = f"blocks.{layer_index}.attn.hook_pattern"
    
    # Get the tensor: shape [batch, n_heads, seq_len, seq_len]
    attention_pattern = cache[layer_name]
    
    # Squeeze batch dimension (batch=1) -> [n_heads, seq_len, seq_len]
    attention_pattern = attention_pattern.squeeze(0)
    
    # Ensure detached and on cpu
    return attention_pattern.detach().cpu().numpy()