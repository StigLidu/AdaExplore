import argparse
import asyncio
import statistics
import time

import aiohttp


ORIGINAL_MODEL = """
import torch
import torch.nn as nn

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(10, 10).cuda()]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)
"""


CUSTOM_MODEL = """
import torch
import torch.nn as nn

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(10, 10).cuda()]

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)
"""


async def fetch_json(session: aiohttp.ClientSession, method: str, url: str, **kwargs):
    async with session.request(method, url, **kwargs) as response:
        try:
            payload = await response.json()
        except Exception:
            payload = {"error": await response.text()}
        return response.status, payload


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    request_id: int,
    semaphore: asyncio.Semaphore,
):
    payload = {
        "original_model_src": ORIGINAL_MODEL,
        "custom_model_src": CUSTOM_MODEL,
        "num_correct_trials": 1,
        "measure_performance": False,
        "verbose": False,
    }

    async with semaphore:
        start = time.time()
        status, result = await fetch_json(
            session,
            "POST",
            f"{base_url}/evaluate",
            json=payload,
        )
        return {
            "request_id": request_id,
            "status": status,
            "elapsed": time.time() - start,
            "success": result.get("success", False),
            "queue_time": float(result.get("queue_time", 0.0) or 0.0),
            "execution_time": float(result.get("execution_time", 0.0) or 0.0),
            "error": result.get("error"),
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:12017")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        health_status, health_payload = await fetch_json(session, "GET", f"{args.url}/health")
        if health_status != 200:
            raise SystemExit(f"health check failed: {health_status} {health_payload}")

        print("health:", health_payload.get("status"), "cuda:", health_payload.get("cuda_available"))

        before_status, before_payload = await fetch_json(session, "GET", f"{args.url}/queue/status")
        if before_status == 200:
            print("queue before:", before_payload)

        started = time.time()
        results = await asyncio.gather(
            *[
                send_request(session, args.url, request_id=i, semaphore=semaphore)
                for i in range(args.requests)
            ]
        )
        total_elapsed = time.time() - started

        after_status, after_payload = await fetch_json(session, "GET", f"{args.url}/queue/status")
        if after_status == 200:
            print("queue after:", after_payload)

    success = [item for item in results if item["success"]]
    failed = [item for item in results if not item["success"]]

    print(f"requests={len(results)} success={len(success)} failed={len(failed)} total_time={total_elapsed:.2f}s")

    if success:
        queue_times = [item["queue_time"] for item in success]
        exec_times = [item["execution_time"] for item in success]
        print(
            "queue_time mean/median/max:",
            f"{statistics.mean(queue_times):.3f}",
            f"{statistics.median(queue_times):.3f}",
            f"{max(queue_times):.3f}",
        )
        print(
            "execution_time mean/median/max:",
            f"{statistics.mean(exec_times):.3f}",
            f"{statistics.median(exec_times):.3f}",
            f"{max(exec_times):.3f}",
        )

    if failed:
        print("failed requests:")
        for item in failed:
            print(item["request_id"], item["status"], item["error"])


if __name__ == "__main__":
    asyncio.run(main())
