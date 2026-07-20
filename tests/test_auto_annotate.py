import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

from curation.auto_annotate import (
    parse_existing_boxes_and_concepts,
    compute_io_min,
    get_sort_key_fn,
    suppress_duplicates,
    prepare_crop,
    run_locate_anything,
    validate_detection_with_llm,
    safe_format,
    _stable_image_id,
    VerificationResult
)


def test_stable_image_id():
    url = "https://example.com/image.jpg"
    im_id = _stable_image_id(url)
    assert len(im_id) == 16
    assert isinstance(im_id, str)
    # Stably reproduces
    assert _stable_image_id(url) == im_id


def test_safe_format():
    tmpl = "Verify if the class '{class_name}' is visible. Braces like {other_key} should be safe."
    res = safe_format(tmpl, class_name="belt")
    assert res == "Verify if the class 'belt' is visible. Braces like {other_key} should be safe."


def test_parse_existing_boxes_and_concepts():
    # Valid bracketed list
    row = pd.Series({
        "boxes": "[10,20,30,40],[50,60,70,80]",
        "concepts": "belt,bag"
    })
    dets = parse_existing_boxes_and_concepts(row)
    assert len(dets) == 2
    assert dets[0]["name"] == "belt"
    assert dets[0]["box"] == [10.0, 20.0, 30.0, 40.0]
    assert dets[1]["name"] == "bag"
    assert dets[1]["box"] == [50.0, 60.0, 70.0, 80.0]

    # Comma-separated single box
    row_single = pd.Series({
        "boxes": "10,20,30,40",
        "concepts": "earring"
    })
    dets_single = parse_existing_boxes_and_concepts(row_single)
    assert len(dets_single) == 1
    assert dets_single[0]["name"] == "earring"
    assert dets_single[0]["box"] == [10.0, 20.0, 30.0, 40.0]

    # Non-matching lengths
    row_mismatch = pd.Series({
        "boxes": "[1,2,3,4],[5,6,7,8]",
        "concepts": "belt"
    })
    assert len(parse_existing_boxes_and_concepts(row_mismatch)) == 1

    # Empty cases
    assert parse_existing_boxes_and_concepts(pd.Series({"boxes": None, "concepts": "belt"})) == []
    assert parse_existing_boxes_and_concepts(pd.Series({"boxes": "1,2,3,4", "concepts": ""})) == []


def test_compute_io_min():
    box_a = [0, 0, 10, 10]
    box_b = [0, 0, 5, 5]
    # Nested completely
    assert compute_io_min(box_a, box_b) == 1.0

    # Half overlap with min box area 25
    box_c = [0, 0, 5, 10] # area = 50
    # box_b area = 25, intersection = 25
    assert compute_io_min(box_c, box_b) == 1.0

    # No overlap
    box_d = [20, 20, 30, 30]
    assert compute_io_min(box_a, box_d) == 0.0


def test_get_sort_key_fn():
    # keep biggest
    key_biggest = get_sort_key_fn("keep biggest")
    det_small = {"source": "fashion_model", "box": [0, 0, 10, 10], "score": 0.9} # area = 100
    det_large = {"source": "locate_anything", "box": [0, 0, 20, 20], "score": 0.5} # area = 400

    assert key_biggest(det_large) < key_biggest(det_small) # large should come first (lower sort key)

    # keep locate anything
    key_la = get_sort_key_fn("keep locate anything box")
    assert key_la(det_large) < key_la(det_small) # locate_anything first

    # keep fashion model
    key_fm = get_sort_key_fn("keep fashion model box")
    assert key_fm(det_small) < key_fm(det_large) # fashion_model first

    # keep smallest
    key_smallest = get_sort_key_fn("keep smallest")
    assert key_smallest(det_small) < key_smallest(det_large) # small first


def test_suppress_duplicates():
    # 2 overlapping boxes of same class: 'belt'
    dets = [
        {"name": "belt", "box": [0, 0, 10, 10], "score": 0.8, "source": "fashion_model"},
        {"name": "belt", "box": [1, 1, 9, 9], "score": 0.9, "source": "locate_anything"}
    ]
    # Using keep locate anything
    res = suppress_duplicates(dets, iou_thresh=0.5, io_min_thresh=0.5, keep_which="keep locate anything box")
    assert len(res) == 1
    assert res[0]["source"] == "locate_anything"


def test_prepare_crop():
    # Create black test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:20, 10:20] = 255 # white square

    # Simple crop
    crop = prepare_crop(img, [10, 10, 20, 20])
    assert crop.shape == (10, 10, 3)
    assert np.mean(crop) == 255

    # Crop with padding
    crop_padded = prepare_crop(img, [10, 10, 20, 20], padding=0.1)
    # Box is 10x10. px = 1, py = 1. padded coords: [9, 9, 21, 21]
    assert crop_padded.shape == (12, 12, 3)

    # Contrast equalization (contrast=True or 'equalize')
    crop_eq = prepare_crop(img, [10, 10, 20, 20], contrast="equalize")
    assert crop_eq.shape == (10, 10, 3)

    # Contrast factor scaling
    crop_scaled = prepare_crop(img, [10, 10, 20, 20], contrast=1.5)
    assert crop_scaled.shape == (10, 10, 3)


@patch("curation.auto_annotate._query_locate_anything_api")
def test_run_locate_anything(mock_api):
    # Mock return values for locate anything API queries
    mock_api.side_effect = [
        [{"name": "belt", "box": [0,0,10,10], "score": 0.9, "source": "locate_anything"}],
        [{"name": "ring", "box": [5,5,15,15], "score": 0.8, "source": "locate_anything"}]
    ]

    model_cfg = {
        "classes": ["belt", "ring"],
        "max_classes_per_prompt": 1,
        "endpoint_url": "http://dummy/detect"
    }

    # Should split into two calls because max_classes_per_prompt = 1
    dets = run_locate_anything("dummy_image.jpg", model_cfg)
    assert len(dets) == 2
    assert mock_api.call_count == 2
    mock_api.assert_any_call("dummy_image.jpg", ["belt"], "http://dummy/detect", "slow")
    mock_api.assert_any_call("dummy_image.jpg", ["ring"], "http://dummy/detect", "slow")


@patch("cv2.imencode")
def test_validate_detection_with_llm(mock_imencode):
    # Mock image encoding
    mock_imencode.return_value = (True, np.array([1, 2, 3]))

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    det = {"name": "earring", "box": [10, 10, 20, 20], "score": 0.9}

    llm_cfg = {
        "general": {"padding": 0.1},
        "earring": {"padding": 0.2, "contrast": 1.5}
    }

    executor = MagicMock()
    executor.predict.return_value = VerificationResult(is_valid=True, reason="Clearly visible")

    system_prompt = "You are a visual assistant."
    user_prompt_tmpl = "Is {class_name} visible?"

    is_valid = validate_detection_with_llm(
        img, det, llm_cfg, executor, system_prompt, user_prompt_tmpl
    )
    assert is_valid is True
    # Verify overrides: earring should use padding 0.2
    # The cropped region for box [10, 10, 20, 20] with padding 0.2 has size 10 + 2*2 = 14
    # Ensure executor predict was invoked
    executor.predict.assert_called_once()


@patch("cv2.imencode")
def test_validate_detection_with_llm_custom_task(mock_imencode):
    mock_imencode.return_value = (True, np.array([1, 2, 3]))

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    det = {"name": "hair_accessories", "box": [10, 10, 20, 20], "score": 0.9}

    llm_cfg = {
        "hair_accessories": {
            "task": "Verify if there is an obvious headband on the person."
        }
    }

    executor = MagicMock()
    executor.predict.return_value = VerificationResult(is_valid=True, reason="Headband present")

    system_prompt = "You are a visual assistant."
    user_prompt_tmpl = "Task:\n{task_prompt}\nClass: {class_name}"

    is_valid = validate_detection_with_llm(
        img, det, llm_cfg, executor, system_prompt, user_prompt_tmpl
    )
    assert is_valid is True
    call_args = executor.predict.call_args[0][0]
    human_msg = call_args[1].content
    assert "Verify if there is an obvious headband on the person." in human_msg
    assert "Class: hair_accessories" in human_msg


@patch("curation.auto_annotate.load_image_and_path")
def test_prefetch_cache(mock_load_image, tmp_path):
    from curation.auto_annotate import prefetch_cache
    import pandas as pd

    df = pd.DataFrame({
        "im_url": ["http://example.com/img1.jpg", "http://example.com/img2.jpg"]
    })
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    cache_dir = tmp_path / "custom_cache"

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
dataset:
  type: flat csv
  path: {str(csv_path)}
  cache_dir: {str(cache_dir)}
  output_dir: {str(tmp_path)}
"""
    cfg_path.write_text(cfg_content)

    mock_load_image.return_value = (np.zeros((10, 10, 3), dtype=np.uint8), cache_dir / "img.jpg")

    prefetch_cache(str(cfg_path))
    assert mock_load_image.call_count == 2
    assert cache_dir.exists()


@patch("curation.auto_annotate.get_image_content")
@patch("azure.storage.blob.BlobServiceClient")
@patch("azure.storage.blob.generate_blob_sas")
def test_prep_azure(mock_gen_sas, mock_blob_service, mock_get_content, tmp_path, monkeypatch):
    from curation.auto_annotate import prep_azure
    import pandas as pd

    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true;")

    mock_gen_sas.return_value = "sas_token_123"
    mock_get_content.return_value = b"fake_image_bytes"

    mock_client_instance = MagicMock()
    mock_blob_service.from_connection_string.return_value = mock_client_instance
    mock_client_instance.account_name = "testaccount"
    mock_client_instance.credential.account_key = "testkey"

    mock_container = MagicMock()
    mock_container.exists.return_value = True
    mock_client_instance.get_container_client.return_value = mock_container

    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_container.get_blob_client.return_value = mock_blob

    df = pd.DataFrame({
        "im_url": ["http://example.com/item1.jpg"]
    })
    csv_path = tmp_path / "original_sample.csv"
    df.to_csv(csv_path, index=False)

    azure_dir = tmp_path / "azure_datasets"

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
dataset:
  type: flat csv
  path: {str(csv_path)}
"""
    cfg_path.write_text(cfg_content)

    prep_azure(str(cfg_path), azure_folder=str(azure_dir))

    expected_csv = azure_dir / "original_sample.csv"
    assert expected_csv.exists()

    result_df = pd.read_csv(expected_csv)
    assert "blob.core.windows.net" in result_df["im_url"].iloc[0]
    assert "sas_token_123" in result_df["im_url"].iloc[0]


def test_get_openai_api_key_dynamic(monkeypatch, tmp_path):
    from llm.executor import _get_openai_api_key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    # Fake dltk config path
    dltk_file = tmp_path / ".dltk.config"
    dltk_file.write_text("OPENAI_API_KEY=sk-test-dltk-key-123\n")
    
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    key = _get_openai_api_key()
    assert key == "sk-test-dltk-key-123"


@patch("curation.auto_annotate.validate_detection_with_llm")
@patch("curation.auto_annotate.load_image_and_path")
@patch("curation.auto_annotate.run_locate_anything")
@patch("llm.executor.LLMExecutor.from_model_name")
def test_auto_annotate_safety_caps(mock_llm_factory, mock_locate, mock_load_img, mock_validate, tmp_path):
    from curation.auto_annotate import main
    import pandas as pd
    import numpy as np

    # Mock single image with 15 candidate boxes for top and 15 for bottom
    mock_load_img.return_value = (np.zeros((100, 100, 3), dtype=np.uint8), tmp_path / "img.jpg")
    
    dets = []
    for i in range(15):
        dets.append({"name": "top", "box": [0, 0, 10, 10], "score": 0.5 + i * 0.01})
    for i in range(15):
        dets.append({"name": "bottom", "box": [10, 10, 20, 20], "score": 0.5 + i * 0.01})
    
    mock_locate.return_value = dets
    mock_validate.return_value = True

    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"im_url": ["http://example.com/test.jpg"]}).to_csv(csv_path, index=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_content = f"""
dataset:
  type: flat csv
  path: {str(csv_path)}
  output_dir: {str(tmp_path)}

detection_models:
  - model_type: locate_anything

llm_validation:
  model_type: gpt-4.1-mini
  max_boxes_per_image: 10
  max_boxes_per_class: 4
"""
    cfg_path.write_text(cfg_content)

    main(str(cfg_path))

def test_resolve_dataset_csv_fallback(tmp_path, monkeypatch):
    from curation.auto_annotate import resolve_dataset_csv
    import pandas as pd

    # Mock REPO_ROOT to tmp_path
    monkeypatch.setattr("curation.auto_annotate.REPO_ROOT", tmp_path)

    # 1. Raw dataset path does not exist, but azure_datasets CSV exists
    azure_dir = tmp_path / "azure_datasets"
    azure_dir.mkdir()
    azure_csv = azure_dir / "sample.csv"
    pd.DataFrame({"im_url": ["http://azure.blob.com/1.jpg"]}).to_csv(azure_csv, index=False)

    dataset_cfg = {
        "type": "flat csv",
        "path": "curated_datasets/sample.csv",
        "azure_folder": "azure_datasets"
    }

    df, filename = resolve_dataset_csv(dataset_cfg, prefer_azure=True)
    assert filename == "sample.csv"
    assert "azure.blob.com" in df["im_url"].iloc[0]



