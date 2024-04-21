"""
This module provides pydantic models to API endpoints.
"""
from pydantic import BaseModel


class OCRRequest(BaseModel):
    """
    Request model for OCR endpoint.
    """

    signed_url: str


class QueryData(BaseModel):
    """
    Request model for query endpoint.
    """

    file_hash: str
    query: str


class UploadResponse(BaseModel):
    """
    Response model for file upload endpoint.
    """

    status: str
    data: list


class TaskResponse(BaseModel):
    """
    Response model for task status endpoint.
    """

    status: str
    data: dict


class ErrorResponse(BaseModel):
    """
    Response model for error messages.
    """

    status: str
    message: str


class QueryResponse(BaseModel):
    """
    Response model for query result.
    """

    status: str
    answer: dict
