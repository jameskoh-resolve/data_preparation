import pytest
from unittest.mock import patch, MagicMock
from curation.sampling.class_sampler import (
    _parse_list_field,
    _stable_image_id,
    _recognize_single_image
)

def test_parse_list_field():
    # None cell
    assert _parse_list_field(None) == []
    
    # List cell
    assert _parse_list_field(["a", "b"]) == ["a", "b"]
    
    # Empty string cell
    assert _parse_list_field("   ") == []
    
    # JSON array string cell
    assert _parse_list_field('["val1", "val2"]') == ["val1", "val2"]
    
    # Python literal list string cell
    assert _parse_list_field("['val1', 'val2']") == ["val1", "val2"]
    
    # Regular string cell fallback
    assert _parse_list_field("regular_string") == ["regular_string"]

def test_stable_image_id():
    img_id = _stable_image_id("test_catalog", "prod_123", "http://example.com/img.jpg")
    assert len(img_id) == 16
    assert all(c in "0123456789abcdef" for c in img_id)
    
    # Should be deterministic
    img_id_2 = _stable_image_id("test_catalog", "prod_123", "http://example.com/img.jpg")
    assert img_id == img_id_2

@patch("curation.sampling.class_sampler.get_image_content")
@patch("requests.post")
def test_recognize_single_image_success(mock_post, mock_get_image_content):
    mock_get_image_content.return_value = b"fake_image_bytes"
    
    # Mock post response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "status": "OK",
        "result": [
            {
                "objects": [
                    {
                        "tags": [
                            {"tag": "dress"},
                            {"tag": "red_dress"}
                        ]
                    }
                ]
            }
        ]
    }
    mock_post.return_value = mock_resp
    
    im_url, tags = _recognize_single_image("http://example.com/img.jpg", "group_a", "v1")
    assert im_url == "http://example.com/img.jpg"
    assert tags == ["dress", "red_dress"]

@patch("curation.sampling.class_sampler.get_image_content")
def test_recognize_single_image_fetch_failure(mock_get_image_content):
    mock_get_image_content.return_value = None
    
    im_url, tags = _recognize_single_image("http://example.com/img.jpg", "group_a", "v1")
    assert im_url == "http://example.com/img.jpg"
    assert tags == []
