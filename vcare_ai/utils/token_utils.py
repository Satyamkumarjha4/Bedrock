def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using a simple word-based approximation
    
    Args:
        text: Input text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Simple approximation: average English word is ~1.3 tokens in most tokenizers
    return int(len(text.split()) * 1.3)