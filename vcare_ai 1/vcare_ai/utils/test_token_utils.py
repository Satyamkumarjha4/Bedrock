import pytest
from vcare_ai.utils.token_utils import estimate_tokens

class TestTokenUtils:
    def test_estimate_tokens_empty_string(self):
        """Test token estimation with empty string"""
        assert estimate_tokens("") == 0
    
    def test_estimate_tokens_single_word(self):
        """Test token estimation with a single word"""
        assert estimate_tokens("hello") == 1
    
    def test_estimate_tokens_multiple_words(self):
        """Test token estimation with multiple words"""
        text = "This is a test of token estimation"
        # 7 words * 1.3 = 9.1, rounded to 9
        assert estimate_tokens(text) == 9
    
    def test_estimate_tokens_with_punctuation(self):
        """Test token estimation with punctuation"""
        text = "Hello, world! How are you today?"
        # 6 words * 1.3 = 7.8, rounded to 7
        assert estimate_tokens(text) == 7 