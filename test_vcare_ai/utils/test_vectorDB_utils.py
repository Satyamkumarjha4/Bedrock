# tests/test_vectorDB_utils.py
import pytest
from unittest.mock import MagicMock, patch, call
from vcare_ai.utils.vectorDB_utils import VectorDBUtils

@pytest.fixture
def mock_db_conn():
    with patch('vcare_ai.utils.vectorDB_utils.psycopg2.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Mock context manager for cursor
        mock_cursor = MagicMock()
        mock_enter = MagicMock()
        mock_enter.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value = mock_enter
        
        yield mock_conn, mock_cursor

def test_db_initialization(mock_db_conn):
    mock_conn, mock_cursor = mock_db_conn
    
    db = VectorDBUtils()
    
    # Verify database initialization calls
    calls = mock_cursor.execute.call_args_list
    assert len(calls) >= 3
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in str(call) for call in calls)
    assert any("CREATE TABLE IF NOT EXISTS food_nutrients" in str(call) for call in calls)

@patch('vcare_ai.utils.vectorDB_utils.SentenceTransformer')
def test_generate_embeddings(mock_model, mock_db_conn):
    # Setup mock embedding model
    mock_encoder = MagicMock()
    mock_model.return_value = mock_encoder
    mock_encoder.encode.return_value = [[0.1, 0.2, 0.3]]  # Return list of embeddings
    
    db = VectorDBUtils()
    result = db.generate_embeddings("test")
    
    # Verify the result is the first embedding array
    assert len(result) == 3
    assert result[0] == 0.1
    assert result[1] == 0.2
    assert result[2] == 0.3

def test_search_similar(mock_db_conn):
    mock_conn, mock_cursor = mock_db_conn
    # Setup mock results
    mock_cursor.fetchall.return_value = [
        ('chicken', {'proteins': 20}, 0.9)
    ]
    
    db = VectorDBUtils()
    results = db.search_similar("chkn")
    
    # Verify results are properly formatted
    assert len(results) == 1
    assert results[0]['food_name'] == "chicken"
    assert results[0]['nutrients']['proteins'] == 20
    assert results[0]['similarity'] == 0.9

def test_get_nutrient_data(mock_db_conn):
    mock_conn, mock_cursor = mock_db_conn
    # Setup mock results
    mock_cursor.fetchone.return_value = ("beef", {"proteins": 25}, 0.85)
    
    db = VectorDBUtils()
    result = db.get_nutrient_data("beef")
    
    # Verify result is properly formatted
    assert result['food_name'] == "beef"
    assert result['nutrients']['proteins'] == 25
    assert result['similarity'] == 0.85