from qrafti import RAG_PATH
from utils import BENCHMARKS_RAG, CHARACTERISTICS_RAG
from rag import RAG
from server_utils import query_rag

import json

if __name__ == "__main__":

    if True:  # Test RAG
        char_rag = RAG(CHARACTERISTICS_RAG, out_dir=RAG_PATH).load()
        bench_rag = RAG(BENCHMARKS_RAG, out_dir=RAG_PATH).load()
        print(json.dumps(query_rag("excess monthly stock returns", rag=char_rag), indent=2))        
#        print(json.dumps(query_rag("quarterly total assets", rag=char_rag), indent=2))
#        print(json.dumps(query_rag('HML', rag=bench_rag), indent=2))
#        print(json.dumps(query_rag('12 month price momentum', rag=char_rag), indent=2))
