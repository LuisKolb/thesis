from typing import List
import pandas as pd

from lkae.retrieval.retrieve import EvidenceRetriever
from lkae.utils.data_loading import AuredDataset, AuthorityPost

from pyterrier.batchretrieve import BatchRetrieve
from pyterrier.index import IndexingType, DFIndexer

from pyterrier.io import write_results, read_results, read_qrels
from pyterrier.pipelines import Evaluate
from ir_measures import R,P,MAP

import logging
logger = logging.getLogger(__name__)

class TerrierRetriever(EvidenceRetriever):
    def __init__(self, k, filename=""):
        super().__init__(k)
        # init pyterrier
        import pyterrier as pt
        if not pt.started():
            pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

    def retrieve(self, rumor_id: str, claim: str, timeline: List[AuthorityPost], **kwargs) -> List:
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
            ranked_results.append([row.qid, row.docno, int(row.rank)+1, row.score]) 

        return ranked_results


    def retrieve_ds(self, dataset: AuredDataset, filename: str):
        # clear output file since we only append later
        if not filename.endswith(".trec.txt"):
            raise ValueError("filename does not end with .trec.txt")
        
        with open(filename, "w"): pass

        for i, entry in enumerate(dataset[:]):
            data = []
            rumor_id = entry['id']
            claim = entry['rumor']
            timeline = entry['timeline']

            for post in timeline:
                data.append(
                    {
                        "qid": rumor_id.strip(), 
                        "query": "".join([c if c.isalnum() else " " for c in claim]).strip(), # clean claim, keep only alphanumeric chars
                        "docno": post.post_id.strip(),
                        "text": post.text.strip()
                    }
                )

            df = pd.DataFrame(data, columns=["qid", "query", "docno", "text"])

            indexref = DFIndexer(index_path="", type=IndexingType.MEMORY).index(df["text"], df)

            bm25 = BatchRetrieve(indexref, wmodel="BM25", controls={"termpipelines": "Stopwords,PorterStemmer"}, metadata=["docno", "text"])
            pl2 = BatchRetrieve(indexref, wmodel="PL2", controls={"termpipelines": "Stopwords,PorterStemmer"}, metadata=["docno", "text"])
            pipeline = (bm25) >> (pl2 % 5)
            
            rtr = pipeline(data)
            write_results(rtr, filename, "trec", append=True)
