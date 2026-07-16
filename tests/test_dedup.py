import pytest
import pandas as pd
from unittest.mock import patch
from io import BytesIO
from PIL import Image
from utils.dedup_image import deduplicate_catalog_by_image


def create_dummy_image_bytes(color: str) -> bytes:
    img = Image.new("RGB", (1, 1), color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@patch("utils.vis_image.get_image_content")
def test_deduplicate_catalog_by_image(mock_get_image_content):
    img_a = create_dummy_image_bytes("red")
    img_b = create_dummy_image_bytes("blue")
    
    # Mocking get_image_content to return identical contents for duplicate images
    def side_effect(url, timeout=10):
        if "img1" in url or "img3" in url:
            return img_a  # img1 and img3 are duplicate images
        return img_b  # img2 is unique
        
    mock_get_image_content.side_effect = side_effect
    
    df = pd.DataFrame({
        "im_url": [
            "http://example.com/img1.jpg", 
            "http://example.com/img2.jpg", 
            "http://example.com/img3.jpg"
        ]
    })
    
    deduped = deduplicate_catalog_by_image(df, url_col="im_url", max_workers=2)
    
    assert len(deduped) == 2
    assert list(deduped["im_url"]) == ["http://example.com/img1.jpg", "http://example.com/img2.jpg"]
