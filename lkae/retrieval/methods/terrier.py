from typing import List
import pandas as pd

from pyterrier.batchretrieve import BatchRetrieve
from pyterrier.index import IndexingType, DFIndexer

from lkae.retrieval.types import EvidenceRetrieverResult
from lkae.retrieval.retrieve import EvidenceRetriever
from lkae.utils.data_loading import AuredDataset, AuthorityPost

import logging
logger = logging.getLogger(__name__)


class TerrierRetriever(EvidenceRetriever):
    def __init__(self, retriever_k, filename="", **kwargs):
        super().__init__(retriever_k)

        # init pyterrier
        import pyterrier as pt
        if not pt.started():
            pt.init()#boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

    def retrieve(self, rumor_id: str, claim: str, timeline: List[AuthorityPost], **kwargs) -> List[EvidenceRetrieverResult]:
        logger.info(f"retrieving documents for rumor_id: {rumor_id}")
        
        # do retrieval "on-the-fly"
        data = []

        # construct a pandas dataframe to pass into terrier
        for post in timeline:
            data.append(
                {
                    "qid": rumor_id.strip(), 
                    # clean claim, keep only alphanumeric chars, or risk tripping up stupid pyterrier grrr
                    "query": "".join([c if c.isalnum() else " " for c in claim]).strip(), 
                    "docno": post.post_id.strip(),
                    "text": post.text.strip()
                }
            )

        df = pd.DataFrame(data, columns=["qid", "query", "docno", "text"])

        # build index from dataframe containing claim+evidence texts for a single rumor_id
        indexref = DFIndexer(index_path="", type=IndexingType.MEMORY).index(df["text"], df)

        # construct a retriever to access the index - to find out the best method, use pt.Experiments and patience :)
        # metadata is optional here
        bm25 = BatchRetrieve(indexref, wmodel="BM25", controls={"termpipelines": "Stopwords,PorterStemmer"}, metadata=["docno", "text"])
        pl2 = BatchRetrieve(indexref, wmodel="PL2", controls={"termpipelines": "Stopwords,PorterStemmer"}, metadata=["docno", "text"])
        pipeline = (bm25 % 10) >> (pl2 % 5)
        
        # results are returned in form of a pandas dataframe
        rtr_df = pipeline(data)

        # pull out single values to conform to the base class interface...
        ranked_results = []
        for row in rtr_df.itertuples():
            assert(rumor_id == row.qid) # should be the same
            ranked_results.append(EvidenceRetrieverResult(str(row.qid), str(row.docno), int(row.rank)+1, float(row.score)))

        return ranked_results