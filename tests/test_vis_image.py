import os
import re
import pytest
import tempfile
from io import BytesIO
from unittest.mock import patch, MagicMock
from PIL import Image

from utils.vis_image import (
    _get_vis_read_url,
    read_vis_atomic,
    get_image_content,
    image_to_base64
)

def test_get_vis_read_url():
    prod_url = _get_vis_read_url("production", "abcdef1234567890abcdef1234567890")
    staging_url = _get_vis_read_url("staging", "abcdef1234567890abcdef1234567890")
    
    assert prod_url == "https://vis.data.visenze.com/v1/image/original/abcdef1234567890abcdef1234567890"
    assert staging_url == "https://staging-vis.data.visenze.com/v1/image/original/abcdef1234567890abcdef1234567890"

def test_read_vis_atomic_validation():
    # Invalid size/characters in md5
    with pytest.raises(ValueError):
        read_vis_atomic("invalid_md5")

def test_read_vis_flat_cache_hit(temp_cache_dir):
    md5 = "12345678901234567890123456789012"
    flat_path = temp_cache_dir / md5
    
    dummy_data = b"flat_cache_data"
    flat_path.write_bytes(dummy_data)
    
    result = read_vis_atomic(md5, cache_folder=temp_cache_dir)
    assert result == dummy_data

def test_read_vis_sharded_cache_hit(temp_cache_dir):
    md5 = "12345678901234567890123456789012"
    shard_dir = temp_cache_dir / md5[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / md5
    
    dummy_data = b"sharded_cache_data"
    shard_path.write_bytes(dummy_data)
    
    result = read_vis_atomic(md5, cache_folder=temp_cache_dir)
    assert result == dummy_data

@patch("urllib.request.urlopen")
def test_read_vis_atomic_download_success(mock_urlopen, temp_cache_dir):
    md5 = "12345678901234567890123456789012"
    dummy_data = b"downloaded_data"
    
    # Mock urllib response
    mock_resp = MagicMock()
    mock_resp.read.return_value = dummy_data
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp
    
    result = read_vis_atomic(md5, cache_folder=temp_cache_dir)
    assert result == dummy_data
    
    # Verify shard file is written
    shard_path = temp_cache_dir / md5[:2] / md5
    assert shard_path.exists()
    assert shard_path.read_bytes() == dummy_data

def test_get_image_content_local_file(tmp_path):
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    
    content = get_image_content(str(img_path))
    assert content is not None
    assert len(content) > 0

def test_image_to_base64():
    img = Image.new("RGB", (100, 100), color="blue")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    b64_str = image_to_base64(BytesIO(img_bytes))
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0
