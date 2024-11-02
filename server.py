import argparse
import sys
import grpc
from concurrent import futures
import torch
import server_pb2
import server_pb2_grpc
from multiprocessing.connection import Listener
import os.path
import json
import numpy as np

import pathlib
import time
from dotenv import load_dotenv
import psutil

load_dotenv()

sys.path.append(str(pathlib.Path(__file__).parent.resolve()) + "/ColBERT")
sys.path.append(str(pathlib.Path(__file__).parent.resolve()) + "/splade_server")

from colbert import Searcher
from colbert.data import Queries
from collections import defaultdict
import requests
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig
from dotenv import load_dotenv
import pathlib
from transformers import AutoTokenizer
from colbert.modeling.base_colbert import BaseColBERT
import ast

import splade_pb2
import splade_pb2_grpc

class ColBERT(torch.nn.Module):
    def __init__(self, lm, linear):
        super().__init__()
        self.lm = lm
        self.linear = linear
    def forward(self, input_ids, attention_mask):
        return self.linear(self.lm(input_ids, attention_mask)[0])


class ColBERTServer(server_pb2_grpc.ServerServicer):
    def __init__(self, num_workers, index, mmap):
        self.threads = num_workers
        self.suffix = "" if not mmap else ".mmap"
        
        if index == "wiki":
            self.index_name = "wiki.2018.latest"
        elif index == "msmarco":
            self.index_name = "msmarco.nbits=2.latest"
        else:
            self.index_name = "lifestyle.dev.nbits=2.latest"
        
        self.multiplier = 250 if index == "wiki" else 500
        self.index_name += self.suffix
        self.prefix = os.environ["DATA_PATH"]

        channel = grpc.insecure_channel('localhost:50060')
        self.splade_stub = splade_pb2_grpc.SpladeStub(channel)

        self.colbert_query_encoder_config = {
            "max_length": 32,
            "q_marker_token_id": 1,
            "mask_token_id": 103,
            "self.cls_token": 101,
            "pad_token_id": 0,
            "device": "cpu",
        }

        self.colbert_results = []
        self.pisa_results = []

        checkpoint_path = "colbert-ir/colbertv2.0"

        self.colbert_search_config = ColBERTConfig(
            index_root=os.path.join(os.environ["DATA_PATH"], "indexes"),
            experiment=self.index_name,
            load_collection_with_mmap=True,
            load_index_with_mmap=mmap,
        )

        process = psutil.Process()
        mem1 = process.memory_info().rss
        self.colbert_searcher = Searcher(
            index=self.index_name,
            checkpoint=checkpoint_path,
            config=self.colbert_search_config,
        )
        
        print(f"MMAP: {mmap}, Index size: {(process.memory_info().rss - mem1) / 1024}")

    def dump(self):
        if self.pisa_results:
            pisa_file = open("ranking_pisa.tsv", "w")
            pisa_file.write("\n".join(["\t".join(x) for x in sorted(self.pisa_results, key=lambda x: (int(x[0]), int(x[2])))])) 
            pisa_file.close()
            self.pisa_results = []
        
        if self.colbert_results:
            colbert_file = open("ranking_colbert.tsv", "w")
            colbert_file.write("\n".join(["\t".join(x) for x in sorted(self.colbert_results, key=lambda x: (int(x[0]), int(x[2])))])) 
            colbert_file.close()
            self.colbert_results = []

    def colbert_search(self, Q, pids, k=5):
        return self.colbert_searcher.dense_search(Q, k=k, pids=pids)


    def colbert_encode(self, queries):
        return self.colbert_searcher.encode([". " + query for query in queries])


    def convert_dict_to_protobuf(self, input_dict):
        query_result = server_pb2.QueryResult()

        query_result.qid = input_dict["qid"]

        for topk_dict in input_dict["topk"]:
            topk_result = query_result.topk.add()
            topk_result.pid = topk_dict["pid"]
            topk_result.rank = topk_dict["rank"]
            topk_result.score = topk_dict["score"]

        return query_result
    
    def api_serve_query(self, query, qid, k=100):
        t2 = time.time()
        url = 'http://localhost:8080'
        splade_q = self.splade_stub.GenerateQuery(splade_pb2.QueryStr(query=query, multiplier=self.multiplier))
        data = {"query": splade_q.query, "k": 200}
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(url, data=json.dumps(data), headers=headers).text
        response = json.loads(response).get('results', {})
        
        for idx, (key, val) in enumerate(sorted(response.items(), key=lambda x: -float(x[1]))):
            self.pisa_results.append((f"{int(qid)}", f"{int(key)}", f"{int(idx+1)}", f"{float(val)}"))


        docs = np.array([int(x) for x in sorted(response.keys())])
        pisa_score = np.array([float(response[x]) for x in sorted(response.keys())])
        pisa_score = (pisa_score - pisa_score.mean()) / pisa_score.std()

        Q = self.colbert_encode([query])
        pids_, _, scores_ = self.colbert_search(Q, docs, 200)
        
        for idx, (key, val) in enumerate(sorted(zip(pids_, scores_), key=lambda x: -x[1])):
           self.colbert_results.append((f"{int(qid)}", f"{int(key)}", f"{int(idx+1)}", f"{float(val)}"))
        
        scores_ = np.array(scores_)
        scores_ = (scores_ - scores_.mean()) / scores_.std()
        
        combined_scores = {}
        
        for d, v in zip(pids_, scores_):
            combined_scores[d] = 0.7 * v
        
        for d, v in zip(docs, pisa_score):
            combined_scores[d] += 0.3 * v

        sorted_pids = sorted(combined_scores.items(), key=lambda x: -x[1])
        
        top_k = []
        for rank, (pid, score) in enumerate(sorted_pids):
            top_k.append({'pid': pid, 'rank': rank + 1, 'score': score})

        print("Serving time of {}: {}".format(qid, time.time() - t2))
        
        return self.convert_dict_to_protobuf({"qid": qid, "topk": top_k[:k]})


    def api_search_query(self, query, qid, k=100):
        t2 = time.time()
        Q = self.colbert_encode([query])
        pids, ranks, scores = self.colbert_search(Q, None, k)

        top_k = []
        for pid, rank, score in zip(pids, ranks, scores):
            top_k.append({'pid': pid, 'rank': rank, 'score': score})
        top_k = list(sorted(top_k, key=lambda p: (-1 * p['score'], p['pid'])))

        combined_scores = {}
        for d, v in zip(pids, scores):
            combined_scores[d] = v
        
        for idx, (key, val) in enumerate(sorted(combined_scores.items(), key=lambda x: -x[1])):
            self.colbert_results.append((f"{int(qid)}", f"{int(key)}", f"{int(idx+1)}", f"{float(val)}"))
        
        print("Searching time of {}: {}".format(qid, time.time() - t2))

        return self.convert_dict_to_protobuf({"qid": qid, "topk": top_k})

    def api_pisa_query(self, query, qid, k=100):
        t2 = time.time()
        splade_q = self.splade_stub.GenerateQuery(splade_pb2.QueryStr(query=query, multiplier=self.multiplier))
        url = 'http://localhost:8080'
        data = {"query": splade_q.query, "k": 200}
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, data=json.dumps(data), headers=headers).text
        response = json.loads(response).get('results', {})

        pids_ = []
        scores_ = []

        for kk, v in response.items():
            pids_.append(int(kk))
            scores_.append(float(v))


        top_k = []
        for pid, rank, score in zip(pids_, range(len(pids_)), scores_):
            top_k.append({'pid': pid, 'rank': rank + 1, 'score': score})
        
        print("Pisa time of {}: {}".format(qid, time.time() - t2))

        return self.convert_dict_to_protobuf({"qid": qid, "topk": top_k[:k]})

    def Search(self, request, context):
        torch.set_num_threads(self.threads)
        return self.api_search_query(request.query, request.qid, request.k)

    def Serve(self, request, context):
        torch.set_num_threads(self.threads)
        return self.api_serve_query(request.query, request.qid, request.k)

    def Pisa(self, request, context):
        torch.set_num_threads(self.threads)
        return self.api_pisa_query(request.query, request.qid, request.k)

    def DumpScores(self, request, context):
        self.dump()
        return server_pb2.Empty()


def serve_ColBERT_server(args):
    connection = None
    if args.run_mode == "driver":
        connection = Listener(('localhost', 50040), authkey=b'password').accept()

    server = grpc.server(futures.ThreadPoolExecutor())
    server_pb2_grpc.add_ServerServicer_to_server(ColBERTServer(args.num_workers, args.index, args.mmap), server)
    listen_addr = '[::]:50050'
    server.add_insecure_port(listen_addr)
    print(f"Starting ColBERT server on {listen_addr}")
    
    if connection is not None:
        connection.send("Done")
        connection.close()

    server.start()
    server.wait_for_termination()
    print("Terminated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for ColBERT')
    parser.add_argument('-w', '--num_workers', type=int, required=True,
                       help='Number of worker threads (torch.num_threads)')
    parser.add_argument('-i', '--index', type=str, required=True, help='Index to run (use "wiki", "msmarco", "lifestyle" to repro the paper, or specify your own index name)')
    parser.add_argument('-m', '--mmap', action="store_true", help='If the index is memory mapped')
    parser.add_argument("-r", "--run_mode", default="server", choices=["server", "driver"], help="Use -r driver while invoking from driver.py")

    args = parser.parse_args()
    serve_ColBERT_server(args)
