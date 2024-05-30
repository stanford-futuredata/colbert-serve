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

import time

load_dotenv()

sys.path.append(os.environ["SPLADE_PATH"])
sys.path.append(str(pathlib.Path(__file__).parent.resolve()) + "/ColBERT")

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
    def __init__(self, num_workers, index):
        self.tag = 0
        self.threads = num_workers
        self.suffix = ""
        self.index_name = "wiki.2018.latest" if index == "wiki" else "lifestyle.dev.nbits=2.latest"
        self.multiplier = 250 if index == "wiki" else 500
        self.index_name += self.suffix
        self.prefix = os.environ["DATA_PATH"]

        channel = grpc.insecure_channel('localhost:50060')
        self.splade_stub = splade_pb2_grpc.SpladeStub(channel)

        colbert_query_encoder_config = {
            "max_length": 32,
            "q_marker_token_id": 1,
            "mask_token_id": 103,
            "self.cls_token": 101,
            "pad_token_id": 0,
            "device": "cpu",
        }

        checkpoint_path = self.prefix + "/msmarco.psg.kldR2.nway64.ib__colbert-400000/"
        self.colbert_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model = BaseColBERT(checkpoint_path)
        lm = base_model.model.LM
        linear = base_model.model.linear
        self.colbert_query_encoder = (
            ColBERT(lm, linear).eval().to(colbert_query_encoder_config["device"])
        )

        self.colbert_search_config = ColBERTConfig(
            index_root=os.path.join(os.environ["DATA_PATH"], "experiments/default/indexes"),
            experiment=self.index_name,
            load_collection_with_mmap=True,
            load_index_with_mmap=bool(os.environ["MMAP"]),
            gpus=0,
        )

        self.colbert_searcher = Searcher(
            index=self.index_name,
            checkpoint=checkpoint_path,
            config=self.colbert_search_config,
        )

    def colbert_search(self, Q, pids, k=5):
        return self.colbert_searcher.dense_search(Q, k=k, pids=pids)


    def colbert_encode(self, queries):
        with torch.no_grad():
            inputs = [". " + query for query in queries]
            tokens = self.colbert_tokenizer(
                inputs,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.colbert_query_encoder_config["max_length"],
            )
            ids, mask = tokens["input_ids"], tokens["attention_mask"]
            ids[:, 1] = self.colbert_query_encoder_config["q_marker_token_id"]
            ids[ids == self.colbert_query_encoder_confignfig["pad_token_id"]] = self.colbert_query_encoder_config["mask_token_id"]
            embeddings = self.colbert_query_encoder(ids, mask)
            return embeddings


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
        gr = torch.tensor([int(x) for x in response.keys()], dtype=torch.int)
        Q = self.colbert_encode([query])
        pids_, ranks_, scores_ = self.colbert_search(Q, gr, 200)
        print("Searching time of {} on node {}: {}".format(qid, self.tag, time.time() - t2))

        scores_sorter = scores_.sort(descending=True)
        pids_, scores_ = pids_[scores_sorter.indices].tolist(), scores_sorter.values.tolist()
        top_k = []
        for pid, rank, score in zip(pids_, range(len(pids_)), scores_):
            top_k.append({'pid': pid, 'rank': rank + 1, 'score': score})

        return self.convert_dict_to_protobuf({"qid": qid, "topk": top_k[:k]})

    def api_search_query(self, query, qid, k=100):
        t2 = time.time()
        pids, ranks, scores = self.searcher.search(query, k)

        print("Searching time of {}: {}".format(qid, time.time() - t2))

        top_k = []
        for pid, rank, score in zip(pids, ranks, scores):
            top_k.append({'pid': pid, 'rank': rank, 'score': score})
        top_k = list(sorted(top_k, key=lambda p: (-1 * p['score'], p['pid'])))

        return self.convert_dict_to_protobuf({"qid": qid, "topk": top_k})

    def api_pisa_query(self, query, qid, k=100):
        t2 = time.time()
        splade_q = self.splade_stub.GenerateQuery(splade_pb2.QueryStr(query=query, multiplier=self.multiplier))
        url = 'http://localhost:8080'
        data = {"query": splade_q.query, "k": 200}
        headers = {'Content-Type': 'application/json'}

        tpost = time.time()
        response = requests.post(url, data=json.dumps(data), headers=headers).text
        response = json.loads(response).get('results', {})

        pids_ = []
        scores_ = []

        for kk, v in response.items():
            pids_.append(int(kk))
            scores_.append(float(v))

        print("Searching time of {} on node {}: {}".format(qid, self.tag, time.time() - t2))

        top_k = []
        for pid, rank, score in zip(pids_, range(len(pids_)), scores_):
            top_k.append({'pid': pid, 'rank': rank + 1, 'score': score})

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


def serve_ColBERT_server(args):
    connection = Listener(('localhost', 50040), authkey=b'password').accept()
    server = grpc.server(futures.ThreadPoolExecutor())
    server_pb2_grpc.add_ServerServicer_to_server(ColBERTServer(args.num_workers, args.index, args.skip_encoding), server)
    listen_addr = '[::]:50050'
    server.add_insecure_port(listen_addr)
    print(f"Starting ColBERT server on {listen_addr}")
    connection.send("Done")
    connection.close()
    server.start()
    server.wait_for_termination()
    print("Terminated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for ColBERT')
    parser.add_argument('-w', '--num_workers', type=int, required=True,
                       help='Number of worker threads per server')
    parser.add_argument('-i', '--index', type=str, choices=["wiki", "lifestyle"],
                        required=True, help='Index to run')

    args = parser.parse_args()
    serve_ColBERT_server(args)
