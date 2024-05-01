"""
This module handles querying Vector DB
and prompting LLM to answer user query.
"""

from main import utils
from main import redis_utils
from main import openai_utils


def merge_overlapping_windows(stored_window: dict):
    """
    Merge overlapping windows in the stored_window dictionary.

    Args:
        stored_window (dict): A dictionary containing window 
            indices as keys and window contents as values.

    Returns:
        dict: A dictionary with merged windows.
    """
    merged_windows = {}
    for idx, focus in stored_window.items():
        if idx not in merged_windows:
            current_focus = set(focus)
            for secondary_idx, secondary_focus in stored_window.items():
                if secondary_idx != idx and secondary_idx not in merged_windows:
                    secondary_focus = set(secondary_focus)
                    if current_focus & secondary_focus:
                        current_focus |= secondary_focus
            merged_windows[idx] = sorted(current_focus)
    return merged_windows


def build_context(merged_windows: dict, doc_utils: redis_utils.RedisUtils, max_context_len: int):
    """
    Build the context string from the merged windows.

    Args:
        merged_windows (dict): A dictionary containing merged window indices 
            as keys and window contents as values.
        doc_utils (redis_utils.RedisUtils): An instance of the RedisUtils class.
        max_context_len (int): The maximum length of the context string.

    Returns:
        str: The context string built from the merged windows.
    """
    context = ""
    for val in merged_windows.values():
        if val:
            context += "".join([doc_utils.get_content_by_id(idx)
                               for idx in val]) + "\n"
            if len(context) > max_context_len:
                break
    return context


async def answer_query(query_dict: dict):
    """
    Perform a query on stored documents based on the provided query string.

    Args:
        query_dict (dict): A dictionary containing the file 
            hash as key and the query string as value.

    Returns:
        str: The response generated by OpenAI based on the query and retrieved context.
    """
    try:
        config = utils.get_config()
        file_hash, query = list(query_dict.items())[0]

        doc_utils = redis_utils.RedisUtils(file_hash, drop=False, create=False)
        query_embedding = openai_utils.get_embedding(query)
        res = doc_utils.search_documents(query, query_embedding)

        if not res:
            utils.logger.info("No relevant context found.")
            return "No relevant context found."

        stored_window = {
            redis_utils.key_to_id(key): doc_utils.get_window_by_id(redis_utils.key_to_id(key))
            for key, _ in res
        }

        merged_windows = merge_overlapping_windows(stored_window)

        prompt_template = "Only use the provided context \
            to answer the query.\nContext: %s\nQuery: %s"
        prompt_len = len(prompt_template % ("", query))
        max_context_len = int(config.get(
            "openai-config", "gpt_context_len")) - prompt_len

        context = build_context(merged_windows, doc_utils, max_context_len)
        prompt = prompt_template % (context, query)

        completion = openai_utils.complete_prompt(prompt)

        utils.logger.info("Prompt length: %d, Answer length: %d.",
                          len(prompt), len(completion))

        return completion

    except Exception as err:
        utils.logger.error("Error occurred, querying %s: %s",
                           str(query_dict), str(err))
        raise
