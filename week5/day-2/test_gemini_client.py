from unittest.mock import MagicMock
from unittest.mock import patch
import pytest
from gemini_client import GeminiClient
 
 
@pytest.fixture
def client():
    with patch("gemini_client.genai.Client"):
        return GeminiClient(
            api_key="dummy-key"
        )
 
 
def test_budget_check(client):
    client.count_tokens = MagicMock(return_value=9000)
    with pytest.raises(ValueError):
        client.generate("Hello")
 
 
def test_successful_generation(client):
    client.count_tokens = MagicMock(return_value=100)
    fake_response = MagicMock()
    fake_response.text = "Hello World"
    usage = MagicMock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    usage.total_token_count = 150
    fake_response.usage_metadata = usage
    client._generate_with_retry = MagicMock(
        return_value=fake_response
    )
    assert (
        client.generate("Hello")
        == "Hello World"
    )
 
 
def test_multimodal(client, tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"fake image")
    client.count_tokens = MagicMock(
        return_value=20
    )
    fake = MagicMock()
    fake.text = "This is a cat."
    usage = MagicMock()
    usage.prompt_token_count = 20
    usage.candidates_token_count = 30
    usage.total_token_count = 50
    fake.usage_metadata = usage
    client._generate_with_retry = MagicMock(
        return_value=fake
    )
    output = client.generate_from_image(
        str(image),
        "What is this?"
    )
    assert output == "This is a cat."
 
 
def test_retry_on_429(client):
    client.count_tokens = MagicMock(return_value=100)
    fake_response = MagicMock()
    fake_response.text = "Hello World"
    usage = MagicMock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    usage.total_token_count = 150
    fake_response.usage_metadata = usage
 
    client.client.models.generate_content = MagicMock(
        side_effect=[
            Exception("429 error"),
            fake_response,
        ]
    )
 
    with patch("gemini_client.time.sleep"):
        result = client.generate("Hello")
 
    assert result == "Hello World"
    assert client.client.models.generate_content.call_count == 2
 