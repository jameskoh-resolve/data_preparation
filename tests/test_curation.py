import pytest
from curation.curate_dataset_v4 import (
    _parse_visible_classes_multiplicity,
    _normalize_box,
    get_box_area,
    compute_io_min,
    _cluster_indices
)

def test_parse_visible_classes_multiplicity():
    # Empty case
    assert _parse_visible_classes_multiplicity("") == {}
    
    # Simple list format
    assert _parse_visible_classes_multiplicity('["shoes", "dress (multiple)"]') == {
        "shoes": "singular",
        "dress": "multiple"
    }

def test_normalize_box():
    # Int rounding and parsing
    assert _normalize_box([10.1, 20.9, 30.5, 40.0]) == [10, 21, 30, 40]
    
    # Invalid length
    with pytest.raises(ValueError):
        _normalize_box([1, 2, 3])

def test_get_box_area():
    assert get_box_area([0, 0, 10, 10]) == 100.0
    assert get_box_area([10, 10, 5, 5]) == 0.0  # Invalid coordinates, max(0.0) handles it

def test_compute_io_min():
    box_a = [0, 0, 10, 10]  # area = 100
    box_b = [0, 0, 5, 5]    # area = 25
    
    # Fully nested, intersection area = 25, min area = 25
    assert compute_io_min(box_a, box_b) == 1.0
    
    # No overlap
    box_c = [20, 20, 30, 30]
    assert compute_io_min(box_a, box_c) == 0.0

def test_cluster_indices():
    # Mocking boxes: box0 overlaps with box1, box2 is separate
    boxes = [
        [0, 0, 10, 10],  # 0
        [1, 1, 11, 11],  # 1 (overlaps heavily with 0)
        [100, 100, 110, 110]  # 2 (far away)
    ]
    clusters = _cluster_indices(boxes, iou_thresh=0.5)
    # Expected clusters: [[0, 1], [2]] or [[1, 0], [2]]
    sorted_clusters = sorted([sorted(c) for c in clusters])
    assert sorted_clusters == [[0, 1], [2]]


def test_load_and_sample_catalogs_v3(tmp_path):
    from curation.curate_dataset_v3 import _load_and_sample_catalogs
    from omegaconf import OmegaConf
    import pandas as pd
    
    df = pd.DataFrame({
        "product_id": ["1", "2"],
        "im_url": ["http://ex.com/1", "http://ex.com/2"],
        "title": [" ", ""],
        "name": ["Name1", "Name2"],
        "description": ["", None],
    })
    
    csv_path = tmp_path / "catalog.csv"
    df.to_csv(csv_path, index=False)
    
    cfg = OmegaConf.create({
        "catalogs": [
            {
                "catalog_csv": str(csv_path),
                "image_url_column": "im_url",
                "category_column": "category"
            }
        ]
    })
    
    loaded_df = _load_and_sample_catalogs(cfg)
    
    assert list(loaded_df["im_name"]) == ["Name1", "Name2"]
    assert list(loaded_df["im_description"]) == ["", ""]


def test_load_and_sample_catalogs_v4(tmp_path):
    from curation.curate_dataset_v4 import _load_and_sample_catalogs
    from omegaconf import OmegaConf
    import pandas as pd
    
    df = pd.DataFrame({
        "product_id": ["1", "2"],
        "im_url": ["http://ex.com/1", "http://ex.com/2"],
        "title": [" ", ""],
        "im_name": ["ImName1", "ImName2"],
        "description": ["", None],
    })
    
    csv_path = tmp_path / "catalog.csv"
    df.to_csv(csv_path, index=False)
    
    cfg = OmegaConf.create({
        "catalogs": [
            {
                "catalog_csv": str(csv_path),
                "image_url_column": "im_url",
                "category_column": "category"
            }
        ]
    })
    
    loaded_df = _load_and_sample_catalogs(cfg)
    
    assert list(loaded_df["im_name"]) == ["ImName1", "ImName2"]
    assert list(loaded_df["im_description"]) == ["", ""]
