import sys

import grpc
import asyncio
import server_pb2
import server_pb2_grpc
import signal
from subprocess import Popen
import argparse
import time
import os
from multiprocessing.connection import Client

from dotenv import load_dotenv
import pathlib

load_dotenv()

print(str(pathlib.Path(__file__).parent.resolve()) + "/ColBERT")
sys.path.append(str(pathlib.Path(__file__).parent.resolve()) + "/ColBERT")

from colbert.data import Queries

def save_rankings(rankings, filename):
    output = []
    for q in rankings:
        for result in q.topk:
            output.append("\t".join([str(x) for x in [q.qid, result.pid, result.rank, result.score]]))

    f = open(filename, "w")
    f.write("\n".join(output))
    f.close()


async def run_request(stub, request, experiment):
    t = time.time()
    if experiment == "search":
        out = await stub.Search(request)
    elif experiment == "pisa":
        out = await stub.Pisa(request)
    else:
        out = await stub.Serve(request)

    return out, time.time() - t


async def run(args):
    queries = Queries(path=f"{os.environ['DATA_PATH']}/{args.index}/questions.tsv")
    qvals = list(queries.items())
    tasks = []

    stub = server_pb2_grpc.ServerStub(grpc.aio.insecure_channel('localhost:50050'))

    inter_request_time = [float(x) for x in open(args.timings).read().split("\n") if x != ""]
    length = len(inter_request_time)

    # Warmup
    for i in range(len(qvals)-100, len(qvals)):
        request = server_pb2.Query(query=qvals[i][1], qid=qvals[i][0], k=100)
        tasks.append(asyncio.ensure_future(run_request(stub, request, args.experiment)))
        await asyncio.sleep(0)

    await asyncio.gather(*tasks)

    await stub.DumpScores(server_pb2.Empty())

    tasks = []
    t = time.time()

    for i in range(min(len(qvals), 1000)):
        request = server_pb2.Query(query=qvals[i][1], qid=qvals[i][0], k=100)
        tasks.append(asyncio.ensure_future(run_request(stub, request,  args.experiment)))
        await asyncio.sleep(inter_request_time[i % length])

    await asyncio.sleep(0)
    ret = list(zip(*await asyncio.gather(*tasks)))

    save_rankings(ret[0], args.ranking_file)

    total_time = str(time.time()-t)

    open(args.output, "w").write("\n".join([str(x) for x in ret[1]]) + f"\nTotal time: {total_time}")
    print(f"Total time for {len(qvals)-100} requests:",  total_time)

    await stub.DumpScores(server_pb2.Empty())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator for ColBERT')
    parser.add_argument('-w', '--num_workers', type=int, required=True,
                       help='Number of worker threads per server')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output file to save results')
    parser.add_argument('-r', '--ranking_file', type=str, required=True,
                        help='Output file to save rankings')
    parser.add_argument('-t', '--timings', type=str, required=True,
                        help='Input file for inter request wait times')
    parser.add_argument('-e', '--experiment', type=str, default="search", choices=["search", "pisa", "serve"],
                        help='search or pisa or serve (pisa + rerank)')
    parser.add_argument('-i', '--index', type=str, choices=["wiki", "msmarco", "lifestyle"],
                        required=True, help='Index to run')
    parser.add_argument('-m', '--mmap', action="store_true", help='If the index is memory mapped')

    args = parser.parse_args()

    arg_str = f"-w {args.num_workers} -i {args.index}" 
    if args.mmap:
        arg_str += " -m"

    process = Popen(["python", "server.py"] + f"{arg_str}".split(" "))

    times = 10
    for i in range(times):
        try:
            connection = Client(('localhost', 50040), authkey=b'password')
            assert connection.recv() == "Done"
            connection.close()
            break
        except ConnectionRefusedError:
            if i == times - 1:
                print("Failed to receive connection for child server. Terminating!")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                sys.exit(-1)
            time.sleep(5)

    asyncio.run(run(args))

    print("Killing processing after completion")
    process.kill()
