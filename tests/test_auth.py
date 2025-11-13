import pytest
import os
from unittest.mock import Mock, patch, MagicMock

# Set up environment variables before any imports
os.environ["SUPABASE_URL"] = "http://test.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "test_service_key"
os.environ["GEMINI_API_KEY"] = "test_gemini_key"

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def mock_supabase():
    """Mock Supabase client for testing."""
    with patch('main.supabase') as mock:
        yield mock


def test_signup_success(mock_supabase):
    """Test successful user signup."""
    mock_session = Mock()
    mock_session.access_token = "test_access_token"
    
    mock_supabase.sign_up.return_value = {
        "session": mock_session,
        "user": {
            "id": "test_user_id",
            "email": "test@example.com"
        }
    }
    
    response = client.post(
        "/signup",
        json={"email": "test@example.com", "password": "password123"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["access_token"] == "test_access_token"
    assert "user" in data


def test_signup_failure(mock_supabase):
    """Test signup failure."""
    mock_supabase.sign_up.side_effect = RuntimeError("Sign up failed: Invalid credentials")
    
    response = client.post(
        "/signup",
        json={"email": "test@example.com", "password": "weak"}
    )
    
    assert response.status_code == 400


def test_login_success(mock_supabase):
    """Test successful user login."""    
    mock_supabase.sign_in.return_value = {
        "session": Mock(access_token="test_access_token"),
        "user": {
            "id": "test_user_id",
            "email": "test@example.com"
        },
        "access_token": "test_access_token"
    }
    
    response = client.post(
        "/login",
        json={"email": "test@example.com", "password": "password123"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["access_token"] == "test_access_token"


def test_login_failure(mock_supabase):
    """Test login with invalid credentials."""
    mock_supabase.sign_in.side_effect = RuntimeError("Sign in failed: Invalid credentials")
    
    response = client.post(
        "/login",
        json={"email": "test@example.com", "password": "wrongpassword"}
    )
    
    assert response.status_code == 401


def test_logout_success(mock_supabase):
    """Test successful logout."""
    mock_supabase.sign_out.return_value = None
    
    response = client.post(
        "/logout",
        headers={"Authorization": "Bearer test_access_token"}
    )
    
    assert response.status_code == 200
    assert response.json() == {"message": "Successfully logged out"}


def test_ask_requires_authentication(mock_supabase):
    """Test that /ask endpoint requires authentication."""
    response = client.post(
        "/ask",
        json={"query": "Who is the top scorer?"}
    )
    
    assert response.status_code == 403  # Forbidden without auth


def test_ask_with_invalid_token(mock_supabase):
    """Test /ask with invalid token."""
    mock_supabase.get_user.side_effect = RuntimeError("Invalid token")
    
    response = client.post(
        "/ask",
        json={"query": "Who is the top scorer?"},
        headers={"Authorization": "Bearer invalid_token"}
    )
    
    assert response.status_code == 401


def test_ask_with_valid_token(mock_supabase):
    """Test /ask with valid authentication."""
    mock_user = {"id": "test_user_id", "email": "test@example.com"}
    mock_supabase.get_user.return_value = mock_user
    
    with patch('main.pipeline') as mock_pipeline:
        mock_pipeline.return_value = {
            "answer": "Cristiano Ronaldo",
            "context": []
        }
        
        response = client.post(
            "/ask",
            json={"query": "Who is the top scorer?"},
            headers={"Authorization": "Bearer valid_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "context" in data
