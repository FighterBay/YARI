"""
This module provides enums.
"""
from enum import IntEnum

# File status enum


class FileStatus(IntEnum):
    """
    Provides enums for FILE_STATUS
    """
    NOT_QUEUED = -2
    ERROR = -1
    PROCESSING_SENTENCE_CHUNKING = 0
    PROCESSING_EMBEDDING_EXTRACTION = 1
    PROCESSED = 2


FILE_STATUS_MESSAGES = {
    FileStatus.NOT_QUEUED: "File not queued.",
    FileStatus.ERROR: "Error in processing.",
    FileStatus.PROCESSING_SENTENCE_CHUNKING: "Processing: sentence chunking.",
    FileStatus.PROCESSING_EMBEDDING_EXTRACTION: "Processing: embedding extraction.",
    FileStatus.PROCESSED: "Processed. Try extraction.",
}
