import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from curation.dataset_generation.upload_datasets import main

@patch("curation.dataset_generation.upload_datasets.parse_args")
@patch("os.path.exists")
@patch("pandas.read_csv")
@patch("pandas.DataFrame.to_csv")
@patch("curation.dataset_generation.upload_datasets.VISUploader")
@patch("curation.dataset_generation.upload_datasets.hv.HydraVisionCreateDataset")
def test_upload_datasets_workflow(
    mock_hv_create_dataset,
    mock_vis_uploader,
    mock_to_csv,
    mock_read_csv,
    mock_exists,
    mock_parse_args
):
    # Mock CLI arguments
    mock_args = MagicMock()
    mock_args.csv_files = ["dataset1.csv"]
    mock_args.dataset_names = ["dataset1"]
    mock_args.image_col = "im_url"
    mock_args.remove_failures = True
    mock_args.skip_mlp = False
    mock_args.type = "CLASSIFICATION"
    mock_parse_args.return_value = mock_args
    
    # Mock existence checks
    mock_exists.return_value = True
    
    # Mock reading the dataset
    input_df = pd.DataFrame({"im_url": ["http://test.com/img1.jpg", "http://test.com/img2.jpg"]})
    mock_read_csv.return_value = input_df
    
    # Mock VISUploader behaviors
    mock_uploader_instance = MagicMock()
    processed_url_df = pd.DataFrame({
        "original_url": ["http://test.com/img1.jpg", "http://test.com/img2.jpg"],
        "vis_uri": ["vis://img1", "vis://img2"],
        "im_url": ["http://vis.hosted.com/img1", "http://vis.hosted.com/img2"]
    })
    mock_uploader_instance.process_dataframes.return_value = [processed_url_df]
    mock_vis_uploader.return_value = mock_uploader_instance
    
    # Mock HydraVision Dataset client
    mock_hv_instance = MagicMock()
    mock_hv_create_dataset.return_value = mock_hv_instance
    
    # Run main logic
    main()
    
    # Verify VISUploader was initialized and executed
    mock_vis_uploader.assert_called_once()
    mock_uploader_instance.process_dataframes.assert_called_once()
    
    # Verify dataset was written to CSV
    mock_to_csv.assert_called_once()
    
    # Verify HydraVision client was initialized and invoked
    mock_hv_create_dataset.assert_called_once_with(
        dataset_name="dataset1",
        dataset_type="CLASSIFICATION",
        update_existing=True
    )
    mock_hv_instance.write_dataframe.assert_called_once()
