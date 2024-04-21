"""
This module provides functions to test various aspects of the modules.
"""
import os
import pytest_dependency
import pytest
from fastapi.testclient import TestClient
from main.app import app, MOCK_OCR_SAVED
from main import models

client = TestClient(app)


@pytest.fixture
def uploaded_files():
    """
    Fixture to upload files and to get signed URL to test OCR endpoint
    """
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file1_path = os.path.join(test_data_dir, "pdf_in_mock.pdf")
    file2_path = os.path.join(test_data_dir, "png_file.png")

    with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
        files = [
            ("files", ("pdf_in_mock.pdf", file1.read(), "application/pdf")),
            ("files", ("png_file.png", file2.read(), "image/png")),
        ]
        response = client.post("/upload", files=files)

    return response

@pytest.mark.dependency()
def test_upload_files_success(uploaded_files):
    """
    Test case for successful file upload.
    """
    assert uploaded_files.status_code == 200
    assert uploaded_files.json()["status"] == "success"
    assert len(uploaded_files.json()["data"]) == 2


def test_upload_files_unsupported_extension():
    """
    Test case for uploading files with unsupported extensions.
    """
    files = [
        ("files", ("file1.txt", b"file1_content", "text/plain")),
    ]
    response = client.post("/upload", files=files)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["data"]) == 0


def test_supported_unsupported_extensions():
    """
    Test case for uploading files with a mix of supported and unsupported extensions.
    """
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file2_path = os.path.join(test_data_dir, "png_file.png")

    with open(file2_path, "rb") as file2:
        files = [
            ("files", ("file1.txt", b"file1_content", "text/plain")),
            ("files", ("png_file.png", file2.read(), "image/png")),
        ]
        response = client.post("/upload", files=files)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["data"]) == 1



@pytest.mark.dependency(depends=["test_upload_files_success"])
def test_do_ocr_success(uploaded_files):
    """
    Test case for successful OCR request.
    """
    signed_url = uploaded_files.json()['data'][0]['signed_url']
    ocr_request = models.OCRRequest(signed_url=signed_url)
    response = client.post("/ocr", json=ocr_request.model_dump())

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "task_id" in response.json()["data"]


def test_do_ocr_invalid_url():
    """
    Test case for OCR request with an invalid URL.
    """
    signed_url = "http://invalid-url"
    ocr_request = models.OCRRequest(signed_url=signed_url)
    response = client.post("/ocr", json=ocr_request.model_dump())

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["message"] == "Error accessing URL"


def test_do_ocr_file_not_in_mock_db():
    """
    Test case for OCR request with a file not present in the mock database.
    """
    signed_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    ocr_request = models.OCRRequest(signed_url=signed_url)
    response = client.post("/ocr", json=ocr_request.model_dump())

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["message"] == "Not in mock DB."

@pytest.mark.skip(reason="Document ingestion wouldn't be finished.")
def test_query_success():
    """
    Test case for successful query.
    """
    file_hash = list(MOCK_OCR_SAVED.keys())[0]
    query = "Sample query"
    query_data = models.QueryData(file_hash=file_hash, query=query)
    response = client.post("/extract", json=query_data.model_dump())

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "answer" in response.json()["answer"]


def test_query_file_not_in_mock_db():
    """
    Test case for query with a file not present in the mock database.
    """
    file_hash = "unknown_hash"
    query = "Sample query"
    query_data = models.QueryData(file_hash=file_hash, query=query)
    response = client.post("/extract", json=query_data.model_dump())

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["message"] == "Not yet queued"
