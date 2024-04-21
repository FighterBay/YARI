"""
This module provides utility functions for interacting with the OpenAI API.
"""

from openai import OpenAI, OpenAIError

from main import utils

def complete_prompt(prompt: str) -> str:
    """
    Completes a given prompt using OpenAI's chat model.

    Args:
        prompt (str): The prompt to be completed.

    Returns:
        str: Completed text.
    """
    try:
        config = utils.get_config()
        client = OpenAI(api_key=config.get("openai-config", "api_key"))

        output = client.chat.completions.create(
            model=config.get("openai-config", "gpt_model"),
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        return output.choices[0].message.content

    except OpenAIError as err:
        utils.logger.error("Error occurred during prompt completion: %s", str(err))
        return ""

    except Exception as err:
        utils.logger.error("Error occurred during prompt completion: %s", str(err))
        return ""


def get_embedding(text: str) -> list:
    """
    Retrieves embedding for a given text using OpenAI's text-embedding model.

    Args:
        text (str): The text to be embedded.

    Returns:
        list: Embedding vector.
    """
    try:
        config = utils.get_config()
        client = OpenAI(api_key=config.get("openai-config", "api_key"))

        response = client.embeddings.create(
            input=text, model=config.get("openai-config", "embedding_model")
        )

        embedding = response.data[0].embedding
        return embedding

    except OpenAIError as err:
        utils.logger.error("Error occurred during embedding retrieval: %s", str(err))
        return []

    except Exception as err:
        utils.logger.error("Error occurred during embedding retrieval: %s", str(err))
        return []
