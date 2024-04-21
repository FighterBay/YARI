# YARI
### Yet another RAG implementation

This project is another RAG implementation and uses BM25 and Cosine Similarity fusion for context retrieval. 

## Table of Contents

- [Setup Instructions](#setup-instructions)
 - [Prerequisites](#prerequisites)
 - [Installation](#installation)
 - [Configuration](#configuration)
- [Usage Instructions](#usage-instructions)
 - [Starting the Services](#starting-the-services)
 - [Stopping the Services](#stopping-the-services)
- [API Documentation](#api-documentation)
 - [Upload Files](#upload-files)
 - [Perform OCR](#perform-ocr)
 - [Extract Information](#extract-information)
- [Testing the API](#testing-the-api)
 - [Upload Files](#test-upload-files)
 - [Perform OCR](#test-perform-ocr)
 - [Extract Information](#test-extract-information)
- [Example and Demo](#example-and-demo)

## Setup Instructions

### Prerequisites

Before running the project, make sure you have the following prerequisites installed:

- Docker
- Docker Compose

### Installation

1. Clone the repository:
```
  git clone ...
```
2. Navigate to the project directory:
```
  cd ...
```

### Configuration

#### config.ini
The project uses a `config.ini` file to store configuration settings. Here's an explanation of each section and its parameters:

- `[minio-config]`:
  - `endpoint`: The URL and port of the Minio server.
  - `access_key`: The access key for authenticating with Minio.
  - `secret_key`: The secret key for authenticating with Minio.
  - `bucket_name`: The name of the bucket to use in Minio for storing files.

- `[redis-config]`:
  - `host`: The hostname or IP address of the Redis server.
  - `port`: The port number of the Redis server.

- `[query-config]`:
  - `window_size`: The size of the window used for sliding window extraction.
  - `top_k`: The number of top matching chunks to consider for extraction.
  - `cosine_link_threshold`: The threshold for cosine similarity to consider chunks as linked.
  - `alpha`: The complimentary weight of the BM25 and Cosine Similarity score for the final score calculation; alpha * bm25_score + (1 - alpha) * cosine_score

- `[document-config]`:
  - `min_chunk_length`: The minimum length of a text chunk.
  - `max_chunk_length`: The maximum length of a text chunk.

- `[openai-config]`:
  - `api_key`: The API key for accessing the OpenAI API.
  - `gpt_model`: The GPT model to use for question-answering.
  - `embedding_model`: The embedding model to use for text embeddings.
  - `embedding_dimension`: The dimension of the text embeddings.
  - `gpt_context_len`: The maximum context length for the GPT model.

Make sure to update the `config.ini` file with your own configuration values before running the project.
Changing just the api_key should be the bare minimum.

#### mock_ocr.ini
- `[MOCK_OCR_SAVED]`: 
    - `MD5_HASH_1`: `FILE_PATH_1`
    - `MD5_HASH_2`: `FILE_PATH_2`
    - `MD5_HASH_3`: `FILE_PATH_3`
    - .
    - .

`FILE_PATH` should be a json file and should be parseable by Python's JSON library. The dictionary structure should be the following:
```
{
    "analyzeResult": {
        "content": ""
    }
}
```
`content`: str, should contain the contents of the file uploaded through the /upload end point as a string.
`MD5_HASH`: str, should be the MD5 hash of the uploaded file, so that it can be mapped to relevant json file and it's contents be ingested.


## Usage Instructions

### Starting the Services

To start the services using Docker Compose, run the following command:
```
docker-compose up --build -d 
```
This will start all the necessary services defined in the `docker-compose.yaml` file, including Redis, Minio, Redis Commander, the FastAPI app, and the Celery worker and build docker image for app and celery worker as well.

### Stopping the Services

To stop the services, run the following command:
```
docker-compose down
```
This will stop and remove all the containers associated with the project.

## API Documentation

### Upload Files

- Endpoint: `/upload`
- Method: POST
- Request Body:
 - `files`: List of files to upload (multipart/form-data)
- Response:
 - `status`: Success or error status
 - `data`: List of uploaded files with their file IDs and signed URLs

### Perform OCR

- Endpoint: `/ocr`
- Method: POST
- Request Body:
 - `signed_url`: Signed URL of the file to perform OCR on
- Response:
 - `status`: Success, error, or processing status
 - `data`: Task ID and message if the file is queued for processing
 - `message`: Error message if an error occurs

### Extract Information

- Endpoint: `/extract`
- Method: POST
- Request Body:
 - `file_hash`: Hash of the file to extract information from
 - `query`: Query to extract information based on
- Response:
 - `status`: Success, error, or processing status
 - `answer`: Extracted information if the file is processed, or processing status if still in progress
 - `message`: Error message if an error occurs

## Testing the API

To test the API endpoints, you can use tools like `curl` or API testing tools like Postman. Here are examples of how to test the endpoints using `curl`:

### Test Upload Files

1. Prepare the files you want to upload (e.g., `file1.pdf`, `file2.png`).

2. Run the following command to upload the files:
```
  curl -X POST -F "files=@file1.pdf" -F "files=@file2.png" http://localhost:8181/upload
```
  This command uploads `file1.pdf` and `file2.png` to the `/upload` endpoint.

3. Check the response to verify the upload status and get the file IDs and signed URLs.

### Test Perform OCR

1. Obtain the signed URL of the file you want to perform OCR on.

2. Run the following command to initiate OCR processing:
```
  curl -X POST -H "Content-Type: application/json" -d '{"signed_url": "https://example.com/file.pdf"}' http://localhost:8181/ocr
```
  Replace `https://example.com/file.pdf` with the actual signed URL of the file.

3. Check the response to verify the OCR processing status and get the task ID if the file is queued for processing.

### Test Extract Information

1. Obtain the MD5 file hash of the processed file you want to extract information from.

2. Run the following command to extract information based on a query:
```
  curl -X POST -H "Content-Type: application/json" -d '{"file_hash": "99ef153b76c24ee4703f3b9e025bab09", "query": "What is the maximum height allowed for buildings?"}' http://localhost:8181/extract
```
  Replace `99ef153b76c24ee4703f3b9e025bab09` with the actual MD5 file hash and provide your desired query.

3. Check the response to get the extracted information if the file is processed, or the processing status if it's still in progress.

Note: Make sure to have the services running locally or replace `http://localhost:8181` with the appropriate URL where your API is hosted.


## Example and Demo

* Make sure YARI is running.
1. demo/demo.py for a streamlit based demo.
* Make sure streamlit is installed before running.
2. examples/example.sh for bash/curl based example.


### Disclaimer
Parts of this readme have been generated by an LLM.


