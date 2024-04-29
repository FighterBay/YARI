"""
This module provides API endpoints to interact with.
"""

import hashlib
import io
from datetime import timedelta

import requests

from fastapi import FastAPI, File, UploadFile, HTTPException
from minio import Minio
from minio.error import S3Error
from requests.exceptions import RequestException

from main import utils, document_processor, query_processor, models, enums
from main.redis_utils import RedisStatus


# Initialize the FastAPI app
app = FastAPI()


# Initialize configparser
config = utils.get_config()

mock_ocr_config = utils.get_mock_ocr_config()

# Minio configuration
MINIO_ENDPOINT = config.get("minio-config", "endpoint")
MINIO_ACCESS_KEY = config.get("minio-config", "access_key")
MINIO_SECRET_KEY = config.get("minio-config", "secret_key")
MINIO_BUCKET_NAME = config.get("minio-config", "bucket_name")
# Allowed file extensions
ALLOWED_EXTENSIONS = ["pdf", "tiff", "png", "jpeg"]

# Mock OCR saved data (only for simulation)
MOCK_OCR_SAVED = mock_ocr_config["MOCK_OCR_SAVED"]

# Initialize Minio client


def create_minio_client():
    """
    Create a Minio client instance.
    Returns:
        minio_client: A minio object
    """

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    return minio_client


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Upload one or more files to the Minio storage.
    """
    uploaded_files = []
    try:
        minio_client = create_minio_client()
        if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
            minio_client.make_bucket(MINIO_BUCKET_NAME)
    except S3Error as err:
        utils.logger.error("Error uploading file: %s", str(err))
        raise HTTPException(status_code=500, detail=str(err)) from err

    for file in files:
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension not in ALLOWED_EXTENSIONS:
            utils.logger.warning(
                "Skipping file %s with unsupported extension.", file.filename
            )
            continue

        file_content = await file.read()
        file_size = len(file_content)
        file_hash = hashlib.md5(file_content).hexdigest()

        unique_filename = f"{file_hash}.{file_extension}"

        try:
            utils.logger.info(
                "Uploading file %s with unique name %s.",
                file.filename,
                unique_filename,
            )
            minio_client.put_object(
                MINIO_BUCKET_NAME,
                unique_filename,
                io.BytesIO(file_content),
                file_size,
                content_type=file.content_type,
            )

            one_hour_delta = timedelta(hours=1)
            signed_url = minio_client.presigned_get_object(
                MINIO_BUCKET_NAME, unique_filename, expires=one_hour_delta
            )
            uploaded_files.append(
                {"file_id": unique_filename, "signed_url": signed_url}
            )
            utils.logger.info("File %s uploaded successfully.", file.filename)
        except S3Error as err:
            utils.logger.error("Error uploading file: %s", str(err))
            raise HTTPException(status_code=500, detail=str(err)) from err

    utils.logger.info("Uploaded %d files successfully.", len(uploaded_files))
    return models.UploadResponse(status="success", data=uploaded_files)


@app.post("/ocr")
async def do_ocr(ocr_request: models.OCRRequest):
    """
    Perform OCR on the provided file URL.
    """
    try:
        response = requests.get(ocr_request.signed_url, timeout=60)
        response.raise_for_status()  # Raise an exception for HTTP errors
        file_content = response.content
        utils.logger.info(
            "Successfully downloaded file from URL: %s", ocr_request.signed_url
        )
    except RequestException as req_err:
        # Handle errors related to invalid or inaccessible URLs
        utils.logger.error("Error accessing URL: %s", str(req_err))
        raise HTTPException(status_code=400, detail="Error accessing URL") from req_err

    file_hash = hashlib.md5(file_content).hexdigest()
    utils.logger.info("Generated file hash: %s", file_hash)

    try:
        redis_status = RedisStatus(file_hash)
        utils.logger.info(
            "Retrieved status for file hash %s from Redis", file_hash
        )
    except Exception as err:
        utils.logger.error(
            "Error fetching %s status from Redis: %s", file_hash, str(err)
        )
        raise HTTPException(status_code=500, detail=str(err)) from err

    processing_status = enums.FileStatus(int(
        redis_status.get_status().get("status",
                                      enums.FileStatus.NOT_QUEUED.value))
    )
    utils.logger.info(
        "File processing status: %s", enums.FILE_STATUS_MESSAGES[processing_status]
    )

    if processing_status not in {enums.FileStatus.NOT_QUEUED,
                                 enums.FileStatus.ERROR,
                                 enums.FileStatus.PROCESSED}:
        raise HTTPException(status_code=400,
                            detail=enums.FILE_STATUS_MESSAGES[processing_status])

    if file_hash in MOCK_OCR_SAVED:
        result = document_processor.document_process.apply_async(
            args=[{file_hash: MOCK_OCR_SAVED[file_hash]}], task_id=file_hash
        )
        utils.logger.info("File %s queued for processing.", result.id)
        return models.TaskResponse(
            status="success",
            data={
                "task_id": result.id,
                "message": f"File {result.id} queued.",
            },
        )

    utils.logger.error("File not found in mock DB.")
    raise HTTPException(status_code=404, detail="Not in mock DB.")


@app.post("/extract")
async def answer_query(query_data: models.QueryData):
    """
    Extract information from the provided file based on the query.
    """

    try:
        redis_status = RedisStatus(query_data.file_hash)
        utils.logger.info(
            "Retrieved RedisStatus object for file hash: %s",
            query_data.file_hash,
        )
    except Exception as err:
        utils.logger.error(
            "Error initiating %s RedisStatus object: %s",
            query_data.file_hash,
            str(err),
        )
        raise HTTPException(status_code=500, detail=str(err)) from err

    processing_status = enums.FileStatus(int(
        redis_status.get_status().get("status",enums.FileStatus.NOT_QUEUED.value))
        )
    utils.logger.info(
        "Processing status for file hash %s: %d: %s",
        query_data.file_hash,
        processing_status,
        enums.FILE_STATUS_MESSAGES[processing_status]
    )

    if processing_status == enums.FileStatus.NOT_QUEUED:
        utils.logger.warning(
            "File %s not yet queued for processing.", query_data.file_hash
        )
        raise HTTPException(status_code=400, detail="Not yet queued")

    if processing_status == enums.FileStatus.PROCESSED:
        try:
            answer = query_processor.answer_query(
                {query_data.file_hash: query_data.query}
            )
            utils.logger.info(
                "Successfully extracted information for query '%s' from file hash %s",
                query_data.query,
                query_data.file_hash,
            )
            return models.QueryResponse(
                status="success", answer={"answer": answer}
            )
        except Exception as err:
            utils.logger.error(
                "Error querying '%s' from file hash %s: %s",
                query_data.query,
                query_data.file_hash,
                str(err),
            )
            raise HTTPException(status_code=500, detail=str(err)) from err
    elif processing_status == enums.FileStatus.ERROR:
        raise HTTPException(status_code=500,
                            detail=str(redis_status.get_status()))

    return models.QueryResponse(
        status="processing", answer=redis_status.get_status()
    )
