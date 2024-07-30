from typing import List, Dict, Union
import json
import os

import logging
logger = logging.getLogger(__name__)

# write retrieval output to file
def write_trec_format_output(filename: str, data: List[List[Union[str, int, float]]], tag: str = "NO_TAG_SPECIFIED") -> None:
    """
    Writes data to a file in the TREC format.

    Parameters:
    - filename (str): The name of the file to write to.
    - data (List[Tuple[str, int, int, float]]): A list of tuples, where each tuple contains:
        - rumor_id (str): The unique ID for the given rumor.
        - authority_tweet_id (int): The unique ID for the authority tweet.
        - rank (int): The rank of the authority tweet ID for that given rumor_id.
        - score (float): The score given by the model for the authority tweet ID.
    - tag (str): The string identifier of the team/model.
    """
    i = 0
    if data:
        with open(filename, 'w') as file:
            for rumor_id, authority_tweet_id, rank, score in data:
                # use " " as separator, pyterrier uses " " as separator by default!! (stupid, please just use "\t")
                line = f"{rumor_id} Q0 {authority_tweet_id} {rank} {score} {tag}\n"
                file.write(line)
                i += 1
        logger.info(f'wrote {i} lines to {filename}')
    else:
        logger.warn('data was empty, nothing was written to disk')

# write verification output to file
def write_jsonlines_from_dicts(filename: Union[str, os.PathLike], dicts: List[Dict]) -> None:
    with open(filename, 'w') as file:
        for item in dicts:
            file.write(f'{json.dumps(item)}\n')