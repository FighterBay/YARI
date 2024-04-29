"""
This module ingests documents to be later 
used to provide context to LLMs for queries.
"""
import json
import numpy as np
from celery import Celery, chord
from main.redis_utils import RedisUtils, RedisStatus
from main.openai_utils import get_embedding
from main import tokenizer, utils, enums


config = utils.get_config()

redis_host = config.get("redis-config", "host")
redis_port = config.get("redis-config", "port")

redis_endpoint = f"{redis_host}:{redis_port}"
celery_app = Celery(
    "tasks",
    broker=f"redis://{redis_endpoint}/2",
    backend=f"redis://{redis_endpoint}/2",
)


@celery_app.task
def process_sentence(idx, sentence, file_hash):
    """
    Process a single sentence and store its information in Redis.

    Args:
        idx (int): The index of the sentence.
        sentence (str): The text of the sentence.
        file_hash (str): The hash of the file being processed.

    Returns:
        int: The index of the processed sentence.
    """
    try:
        sentence_dict = {}
        embedding = get_embedding(sentence)

        if len(embedding) != int(
            config.get("openai-config", "embedding_dimension")
        ):
            return 0

        sentence_dict["embedding"] = np.asarray(embedding)

        doc_utils = RedisUtils(file_hash, False, False)

        sentence_dict["doc_chunk_num"] = idx
        sentence_dict["content"] = sentence
        # Similarity score of the next chunk
        sentence_dict["next_sim_score"] = -1

        doc_utils.insert_sentence(sentence_dict)
        RedisStatus(file_hash).increment_processed()

        return idx
    except Exception as err:
        # Log any exceptions that occur during sentence processing
        utils.logger.error("Error processing sentence %d: %s", idx, str(err))

        # Add to DLQ(dead lette queue) to manage the error later
        return 0


@celery_app.task()
def progress_update_callback(group_result, main_task_id):
    """
    Update the progress status based on the results of the group of tasks.

    Args:
        group_result (List[int]): List of results from the group of tasks.
        main_task_id (str): The ID of the main task.

    Returns:
        int: 1 if all tasks were successful, 0 otherwise.
    """
    try:
        successful = all(result != 0 for result in group_result)

        redis_status = RedisStatus(main_task_id)
        utils.logger.info(
            "Document ingestion task %s: %s", main_task_id, str(successful)
        )
        if successful:
            redis_status.change_status(enums.FileStatus.PROCESSED.value)
            return 1

        # else
        redis_status.change_status(enums.FileStatus.ERROR.value)
        return 0
    except Exception as err:
        # Log any exceptions that occur during progress update
        utils.logger.error(
            "Error updating task %s: %s", main_task_id, str(err)
        )
        return 0


def pre_process_sentences(sentences: list):
    """
    Pre-process sentences by combining and splitting them 
    based on minimum and maximum chunk lengths.

    Args:
        sentences (list): A list of sentences to be pre-processed.

    Returns:
        list: A list of pre-processed sentences, 
        where each sentence is within the specified length range.
    """
    last_sentence = ""
    new_sentences = []
    min_chunk_length = int(config.get("document-config", "min_chunk_length"))
    max_chunk_length = int(config.get("document-config", "max_chunk_length"))

    for sentence in sentences:
        total_sentence = last_sentence + sentence
        total_sentence_len = len(total_sentence)

        if total_sentence_len < min_chunk_length:
            last_sentence += sentence
        else:
            last_sentence = ""

            if total_sentence_len > max_chunk_length:
                chunked_sentences = [
                    total_sentence[idx: idx + max_chunk_length]
                    for idx in range(0, total_sentence_len, max_chunk_length)
                ]
                for sentence_chunk in chunked_sentences:
                    new_sentences.append(sentence_chunk)

                last_sentence_chunk = new_sentences[-1]
                if len(last_sentence_chunk) < min_chunk_length:
                    new_sentences.pop()
                    last_sentence = last_sentence_chunk
            else:
                new_sentences.append(total_sentence)

    if last_sentence:
        new_sentences.append(last_sentence)

    return new_sentences


@celery_app.task(bind=True)
def document_process(self, file_dict: dict):
    """
    Process a document containing Japanese text.

    Args:
        self: The Celery task instance.
        file_dict (dict): A dictionary containing the hash of the file as key and its path as value.

    Returns:
        int: 1 if the document processing was successful, 0 otherwise.
    """
    try:
        file_hash, file_path = next(iter(file_dict.items()))

        if not file_path.endswith(".json"):
            utils.logger.error("Error: File is not a JSON file")
            return 0

        with open(file_path, "r", encoding="utf-8") as file_handle:
            json_obj = json.load(file_handle)

        japanese_text = json_obj.get("analyzeResult", {}).get("content", "")
        if not japanese_text:
            utils.logger.error("Error: Japanese text not found in JSON")
            return 0

        sentences = tokenizer.split_japanese_sentences(japanese_text)

        # Drop index and recreate it
        RedisUtils(file_hash, drop=True, create=True)

        redis_status = RedisStatus(file_hash)
        redis_status.create_status()

        new_sentences = pre_process_sentences(sentences)

        redis_status.set_sentence_count(len(new_sentences))
        sub_tasks = [
            process_sentence.s(idx + 1, sentence, file_hash)
            for idx, sentence in enumerate(new_sentences)
        ]

        chord(sub_tasks)(progress_update_callback.s(self.request.id))

        redis_status.change_status(
            enums.FileStatus.PROCESSING_EMBEDDING_EXTRACTION.value)

        return 1
    except Exception as err:
        # Log any exceptions that occur during progress update
        utils.logger.error("Error processing document: %s", str(err))
        return 0
