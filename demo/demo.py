"""
This module is streamlit demo app to interact with the APIs.
"""
import requests
import streamlit as st
from PIL import Image

# FastAPI endpoint URL
API_URL = "http://localhost:8181"

def upload_files():
    """
    Allows the user to upload multiple files and sends them to the FastAPI endpoint for processing.
    Returns:
        list: A list of dictionaries containing the uploaded file data.
    Raises:
        ValueError: If no files are selected for upload.
    """
    uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
    if uploaded_files:
        files = [("files", (file.name, file.getvalue())) for file in uploaded_files]
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success":
                st.success("Files uploaded successfully!")
                return data["data"]
            else:
                st.error(f"Error uploading files: {data['message']}")
        else:
            st.error(f"Error uploading files: {response.text}")
    else:
        raise ValueError("No files selected for upload.")
    return []

def perform_ocr(file_data: dict):
    """
    Sends a request to the FastAPI endpoint to perform OCR on the specified file.
    Args:
        file_data (dict): A dictionary containing the file data.
    Returns:
        str: The task ID of the OCR task if successful, None otherwise.
    Raises:
        ValueError: If the file data is invalid or missing required fields.
    """
    if not isinstance(file_data, dict) or "signed_url" not in file_data:
        raise ValueError("Invalid file data. Missing required fields.")

    ocr_request = {"signed_url": file_data["signed_url"]}
    response = requests.post(f"{API_URL}/ocr", json=ocr_request)
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "success":
            st.success(f"OCR task queued for file: {file_data['file_id']}")
            return data["data"]["task_id"]
        else:
            st.error(f"Error performing OCR: {data['message']}")
    else:
        st.error(f"Error performing OCR: {response.text}")
    return None

def extract_information(file_hash: str, query: str):
    """
    Sends a request to the FastAPI endpoint to extract information from the specified file based on the provided query.
    Args:
        file_hash (str): The hash of the file to extract information from.
        query (str): The query to extract information based on.
    Raises:
        ValueError: If the file hash or query is empty or invalid.
    """
    if not file_hash or not query:
        raise ValueError("File hash and query cannot be empty.")

    query_data = {"file_hash": file_hash, "query": query}
    response = requests.post(f"{API_URL}/extract", json=query_data)
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "success":
            st.success("Information extracted successfully!")
            st.write(data["answer"])
        elif data["status"] == "processing":
            st.warning("File processing in progress. Please wait.")
        else:
            st.error(f"Error extracting information: {data['message']}")
    else:
        st.error(f"Error extracting information: {response.text}")

def main():
    st.title("Document Processing App")

    # Create pages for upload, OCR, and extract
    pages = {
        "Upload Files": upload_files,
        "Perform OCR": perform_ocr,
        "Extract Information": extract_information,
    }

    # Sidebar navigation
    selected_page = st.sidebar.radio("Select a page", list(pages.keys()))

    # Display the selected page
    if selected_page == "Upload Files":
        try:
            uploaded_files = upload_files()
            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                st.info(uploaded_files)
        except ValueError as e:
            st.warning(str(e))
    elif selected_page == "Perform OCR":
        if "uploaded_files" in st.session_state:
            selected_file = st.selectbox("Select a file", [f["file_id"] for f in st.session_state.uploaded_files])
            selected_file_data = next(f for f in st.session_state.uploaded_files if f["file_id"] == selected_file)
            if st.button("Perform OCR"):
                try:
                    task_id = perform_ocr(selected_file_data)
                    if task_id:
                        st.info(f"OCR task queued with ID: {task_id}")
                except ValueError as e:
                    st.warning(str(e))
        else:
            st.warning("No files uploaded. Please upload files first.")
    elif selected_page == "Extract Information":
        if "uploaded_files" in st.session_state:
            selected_file = st.selectbox("Select a file", [f["file_id"] for f in st.session_state.uploaded_files])
            query = st.text_input("Enter a query")
            if st.button("Extract Information"):
                try:
                    extract_information(selected_file.split(".")[0], query)
                except ValueError as e:
                    st.warning(str(e))
        else:
            st.warning("No files uploaded. Please upload files first.")

if __name__ == "__main__":
    main()