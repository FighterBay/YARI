"""
This module provides utility functions for interacting 
    with the Redis to help with queries and document ingestion.
"""
import redis
import numpy as np

from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition

from main import utils
from main import tokenizer


class RedisStatus:
    """
    Provides a utilities class for fetching status of Document ingestion process.
    """

    def __init__(self, file_hash: str):
        """
        Initialize RedisStatus with the provided file hash.

        Args:
            file_hash (str): The hash of the file.

        Returns:
            None
        """

        self.config = utils.get_config()

        redis_host = self.config.get("redis-config", "host")
        redis_port = int(self.config.get("redis-config", "port"))

        try:
            # Connect to Redis
            self.client = redis.Redis(
                host=redis_host, port=redis_port, db=1, decode_responses=True
            )
        except redis.ConnectionError as err:
            # Log connection error
            utils.logger.error("Failed to connect to Redis: %s", str(err))
            # Initialize client as None
            self.client = None
            raise

        self.file_hash = file_hash

    def __del__(self):
        """
        Destructor to close redis-client
        """
        if self.client is not None:
            self.client.close()

    def get_status(self):
        """
        Get status information for the file hash from Redis.

        Returns:
            dict: A dictionary containing status information.
        """

        try:
            # Attempt to fetch status from Redis
            return self.client.hgetall(self.file_hash)
        except redis.RedisError as err:
            # Log Redis error
            utils.logger.error(
                "Error fetching %s status from Redis: %s",
                self.file_hash,
                str(err),
            )
            return {}

    def create_status(self):
        """
        Create initial status entries in Redis for the file hash.

        Returns:
            None
        """

        try:
            # Pipeline multiple commands for atomic execution
            pipe = self.client.pipeline()
            # Set initial status
            pipe.hset(name=self.file_hash, key="status", value=0)
            # Set processed count to 0
            pipe.hset(name=self.file_hash, key="processed_count", value=0)
            # Set sentence count to 0
            pipe.hset(name=self.file_hash, key="sentence_count", value=0)
            # Execute the pipeline
            pipe.execute()
        except redis.RedisError as err:
            # Log Redis error
            utils.logger.error(
                "Error creating status for %s in Redis: %s",
                self.file_hash,
                str(err),
            )
            raise

    def set_sentence_count(self, value: int):
        """
        Set the sentence count for the file hash in Redis.

        Args:
            value (int): The sentence count.

        Returns:
            int: The result of the hset operation.
        """

        try:
            # Set sentence count
            return self.client.hset(self.file_hash, "sentence_count", value)
        except redis.RedisError as err:
            # Log Redis error
            utils.logger.error(
                "Error setting sentence count for %s in Redis: %s",
                self.file_hash,
                str(err),
            )
            return 0

    def increment_processed(self):
        """
        Increment the processed count for the file hash in Redis.

        Returns:
            int: The result of the hincrby operation.
        """

        try:
            # Increment processed count
            return self.client.hincrby(self.file_hash, "processed_count")
        except redis.RedisError as err:
            # Log Redis error
            utils.logger.error(
                "Error incrementing processed count for %s in Redis: %s",
                self.file_hash,
                str(err),
            )
            return 0

    def change_status(self, new_status: int):
        """
        Change the status of the file hash in Redis.

        Args:
            new_status (int): The new status value.

        Returns:
            int: The result of the hset operation.
        """

        try:
            # Change status
            return self.client.hset(self.file_hash, "status", new_status)
        except redis.RedisError as err:
            # Log Redis error
            utils.logger.error(
                "Error changing status for %s in Redis: %s",
                self.file_hash,
                str(err),
            )
            return 0


class RedisUtils:
    """
    Provides a utilities class for document ingestion.
    """

    def __init__(self, document_name, drop: bool = False, create: bool = False):
        """
        Initialize RedisUtils with the document name.

        Args:
            document_name (str): The name of the document.
            drop (bool, optional): Whether to drop the index. Defaults to False.
            create (bool, optional): Whether to create the index. Defaults to False.

        Returns:
            None
        """

        self.config = utils.get_config()
        redis_host = self.config.get("redis-config", "host")
        redis_port = int(self.config.get("redis-config", "port"))

        try:
            # Connect to Redis
            # Redis tries to decode embeddings as well, hence setting
            # decode_responses=False
            self.client = redis.Redis(
                host=redis_host, port=redis_port, db=0, decode_responses=False
            )
        except redis.ConnectionError as err:
            # Log connection error
            utils.logger.error("Failed to connect to Redis: %s", str(err))
            # Initialize client as None
            self.client = None
            raise

        self.document_name = f"document_{document_name}"
        self.index_name = f"idx:{self.document_name}"

        if self.client is not None:
            if drop:
                try:
                    self.client.ft(self.index_name).dropindex(True)
                except BaseException:
                    pass

            if create:
                self._setup_index()

    def __del__(self):
        """
        Destructor to close redis-client
        """
        if self.client is not None:
            self.client.close()

    def _setup_index(self):
        """
        Setup the index in Redis.

        Returns:
            Search: Redis Search object
        """

        index = self.client.ft(self.index_name)

        try:
            index.info()  # Check if index exists
            return index
        except redis.RedisError as err:
            utils.logger.info(
                    "Index %s: not found. Creating it: %s", self.index_name, err
                )

            try:
                schema = (
                    TextField("content"),
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": int(
                                self.config.get(
                                    "openai-config", "embedding_dimension"
                                )
                            ),
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                )
                return self.client.ft(self.index_name).create_index(
                    schema,
                    definition=IndexDefinition(
                        prefix=[f"{self.document_name}:"]
                    ),
                    stopwords=utils.ja_stop_words,
                )
            except redis.RedisError as second_err:
                # Log index creation error
                utils.logger.error(
                    "Failed to create index %s: %s", self.index_name, str(second_err)
                )
                raise

    def insert_sentence(self, sentence_properties: dict):
        """
        Insert a sentence into Redis.

        Args:
            sentence_properties (dict): A dictionary containing sentence properties.

        Returns:
            None
        """
        # Tokenize Japanese ourself as Redis index doesn't support it directly

        try:
            # Reconstruct it back or can save unprocessed content directly but
            # space requirements will grow(which is better than compute
            # requirements)
            sentence_properties["raw_content"] = sentence_properties["content"]
            sentence_properties["content"] = (
                tokenizer.tokenize_and_join_with_whitespace(
                    sentence_properties["content"]
                )
            )

            pipe = self.client.pipeline()
            doc_name = f"{self.document_name}:{sentence_properties['doc_chunk_num']}"
            pipe.hset(
                name=doc_name,
                key="content",
                value=sentence_properties["content"],
            )
            pipe.hset(
                name=doc_name,
                key="raw_content",
                value=sentence_properties["raw_content"],
            )
            pipe.hset(
                name=doc_name,
                key="next_sim_score",
                value=sentence_properties["next_sim_score"],
            )
            pipe.hset(
                name=doc_name,
                key="embedding",
                value=np.array(
                    sentence_properties["embedding"], dtype=np.float32
                ).tobytes(),
            )
            pipe.execute()
        except Exception as err:
            # Log index creation error
            utils.logger.error(
                "Error while inserting sentence %d to %s: %s",
                sentence_properties["doc_chunk_num"],
                self.index_name,
                str(err),
            )
            raise

    def _custom_bm25_scoring(self, query: str):
        """
        Custom BM25 scoring for the given query.

        Args:
            query (str): The query for which BM25 scoring is performed.

        Returns:
            dict: A dictionary containing combined and normalized BM25 scores for each document.
        """

        try:
            # Tokenize the query and construct a Redis query
            tokenized_query = tokenizer.tokenize_and_construct_redis_query(
                query
            )
            total_tokens_len = len(tokenized_query)

            # If the tokenized query is empty, return an empty dictionary
            if total_tokens_len == 0:
                return {}

            # Initialize a dictionary to store the combined BM25 scores for
            # each document
            combined_bm25_docs = {}

            # Set batch size for paging
            batch_size = 100

            # Iterate over each token in the tokenized query
            for token in tokenized_query:
                # Initialize offset for paging
                offset = 0

                # Iterate until all results are fetched
                while True:
                    # Construct a Redis query to search for the token using
                    # BM25 scoring with paging
                    bm25_query = (
                        Query(f"@content:{token}")
                        .scorer("BM25")
                        .with_scores()
                        .return_fields("id", "score")
                        .paging(offset, batch_size)
                    )

                    # Execute the Redis query and retrieve the search results
                    bm25_res_docs = self.client.ft(self.index_name).search(
                        bm25_query
                    )

                    # If there are no search results, break the loop
                    if not bm25_res_docs.docs:
                        break

                    # Extract the scores from the search results
                    scores = [doc.score for doc in bm25_res_docs.docs]

                    # Calculate the maximum and minimum BM25 scores
                    max_bm25_score = max(scores)
                    min_bm25_score = min(scores)

                    # Calculate the range of BM25 scores
                    max_minus_min_bm25 = max_bm25_score - min_bm25_score

                    # Iterate over each search result document
                    for doc in bm25_res_docs.docs:
                        # Normalize the BM25 score of the document
                        if max_minus_min_bm25 != 0:
                            normalized_score = (
                                doc.score - min_bm25_score
                            ) / max_minus_min_bm25
                        else:
                            normalized_score = (
                                1 if max_minus_min_bm25 == 0 else 0
                            )

                        # Update the combined BM25 score for the document
                        combined_bm25_docs[doc.id] = (
                            combined_bm25_docs.get(doc.id, 0)
                            + normalized_score
                        )

                    # Update offset for the next batch
                    offset += batch_size

            # Normalize the combined BM25 scores by dividing by the total
            # number of tokens
            for doc_id in combined_bm25_docs:
                combined_bm25_docs[doc_id] /= total_tokens_len

            # Return the dictionary containing the combined and normalized BM25
            # scores for each document
            return combined_bm25_docs

        except Exception as err:
            # Error handling
            utils.logger.error(
                "An error occurred during custom BM25 scoring: %s", str(err)
            )
            return {}

    def search_documents(self, query: str, query_embedding: np.ndarray):
        """
        Search for documents using a combination of BM25 scoring and cosine similarity.

        Args:
            query (str): The search query.
            query_embedding (np.ndarray): The embedding vector representation of the query.


        Returns:
            list: A list of tuples containing document IDs and their combined scores.
        """

        try:

            top_k = int(self.config.get("query-config", "top_k"))

            # Perform custom BM25 scoring
            normalized_bm25_res_dict = self._custom_bm25_scoring(query)

            # Prepare query vector for cosine similarity search
            query_vector = np.array(
                query_embedding, dtype=np.float32
            ).tobytes()

            # Perform cosine similarity search using Redis
            cosine_query = (
                Query("*=>[KNN $top_k @embedding $vector AS score]")
                .return_fields("id", "score")
                .dialect(2)
                .paging(0, top_k)
            )
            cosine_res_docs = self.client.ft(self.index_name).search(
                cosine_query, {"vector": query_vector, "top_k": top_k}
            )

            # Convert cosine similarity results to a dictionary
            cosine_res_dict = {
                doc.id: float(doc.score) for doc in cosine_res_docs.docs
            }

            # Combine BM25 and cosine similarity results
            alpha = float(self.config.get("query-config", "alpha"))
            penalize = (
                -0.1
            )  # Penalty for not having any query token in the content
            res_dict = {}
            for key in set(normalized_bm25_res_dict.keys()) | set(
                cosine_res_dict.keys()
            ):
                bm25_score = normalized_bm25_res_dict.get(
                    key, penalize
                )  # penalize
                # 0 if not found
                cosine_score = 1 - cosine_res_dict.get(key, 1)
                res_dict[key] = {
                    "combined": alpha * bm25_score + (1 - alpha) * cosine_score,
                    "bm25": bm25_score,
                    "cosine": cosine_score,
                }

            # Sort the results based on the combined score and return the top-k
            # documents
            sorted_res = sorted(
                res_dict.items(), key=lambda e: e[1]["combined"], reverse=True
            )[:top_k]

            # Logging
            utils.logger.info(
                "Search query: %s. Top %d documents found.",
                query,
                len(sorted_res),
            )

            return sorted_res

        except Exception as err:
            # Error handling
            utils.logger.error(
                "An error occurred during document search: %s", str(err)
            )
            return []

    def get_window_by_id(self, idx: int):
        """
        Get a window of keys centered around a given ID.

        Args:
            idx (int): The center ID around which the window is constructed.

        Returns:
            list: A list of IDs representing the window.
        """

        try:
            window_size = int(self.config.get("query-config", "window_size"))
            cosine_link_threshold = float(self.config.get("query-config", "cosine_link_threshold"))
            center_key = f"{self.document_name}:{idx}"

            window = [center_key]
            for direction in [-1, 1]:
                for i in range(1, window_size + 1):
                    key = f"{self.document_name}:{idx + direction * i}"
                    next_key = f"{self.document_name}:{idx + direction * (i + 1)}"

                    if not self.client.exists(key) or not self.client.exists(next_key):
                        break

                    next_sim_score = self.get_item_by_key(key, ["next_sim_score"])
                    if next_sim_score:
                        next_sim_score = float(next_sim_score[0].decode())
                        if next_sim_score == -1:
                            next_sim_score = self.compute_cosine_similarity(key, next_key)
                            self.set_next_sim_score_by_key(key, next_sim_score)

                        if next_sim_score >= cosine_link_threshold:
                            if direction == -1:
                                window.insert(0, key)
                            else:
                                window.append(key)
                        else:
                            break

            return [key_to_id(key) for key in window]
        except Exception as err:
            utils.logger.error("An error occurred while creating \
                               window for ID %d: %s", id, str(err))
            raise

    def compute_cosine_similarity(self, key1: str, key2: str):
        """
        Compute the cosine similarity between the embeddings of two keys.

        Args:
            key1 (str): The first key.
            key2 (str): The second key.

        Returns:
            float: The cosine similarity between the embeddings of the two keys.
        """
        embedding1 = np.frombuffer(self.get_item_by_key(key1, ["embedding"])[0], dtype=np.float32)
        embedding2 = np.frombuffer(self.get_item_by_key(key2, ["embedding"])[0], dtype=np.float32)
        return utils.cosine_similarity_numba(embedding1, embedding2)

    def get_item_by_key(self, key: str, fields: list):
        """
        Retrieve specific fields from a Redis hash identified by a key.

        Args:
            key (str): The key identifying the Redis hash.
            fields (list): A list of fields to retrieve.

        Returns:
            list: A list containing the values of the specified fields.
        """

        try:
            return_fields = self.client.hmget(key, *fields)
            return [field for field in return_fields if field is not None]
        except Exception as err:
            # Log the error
            utils.logger.error(
                "An error occurred while retrieving item by key %s: %s",
                key,
                str(err),
            )

            raise

    def set_next_sim_score_by_key(self, key: str, next_sim_score: float):
        """
        Set the similarity score against previous chunk for a Redis hash identified by a key.

        Args:
            key (str): The key identifying the Redis hash.
            next_sim_score (float): The similarity score to set.

        Returns:
            int: The result of the hset operation.
        """

        try:
            return self.client.hset(key, "next_sim_score", next_sim_score)
        except Exception as err:
            # Log the error
            utils.logger.error(
                "An error occurred while setting next_sim_score by key %s: %s",
                key,
                str(err),
            )
            return 0

    def get_content_by_key(self, key: str):
        """
        Retrieve the raw content from a Redis hash identified by a key.

        Args:
            key (str): The key identifying the Redis hash.

        Returns:
            str: The raw content.
        """

        try:
            return self.client.hmget(key, "raw_content")[0].decode()
        except Exception as err:
            # Log the error
            utils.logger.error(
                "An error occurred while retrieving content by key %s: %s",
                key,
                str(err),
            )
            # Raise the error
            raise

    def get_content_by_id(self, idx: int):
        """
        Retrieve the raw content from a Redis hash identified by an ID.

        Args:
            idx (int): The ID.

        Returns:
            str: The raw content.
        """

        try:
            key = f"{self.document_name}:{idx}"
            return self.get_content_by_key(key)
        except Exception as err:
            # Log the error
            utils.logger.error(
                "An error occurred while retrieving content by ID %s: %s",
                idx,
                str(err),
            )
            # Raise the error
            raise


def key_to_id(key: str):
    """
    Extract the ID from a key string.

    Args:
        key (str): The key string containing the ID.

    Returns:
        int: The extracted ID.
    """

    try:
        return int(key.split(":")[-1])
    except Exception as err:
        # Log the error
        utils.logger.error(
            "An error occurred while extracting ID from key %s: %s",
            key,
            str(err),
        )
        # Raise the error
        raise
