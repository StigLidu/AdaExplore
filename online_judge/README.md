# Online Judge

This directory contains the minimal remote evaluation service used by HardAgent.

## Files

- `app_with_queue.py`: FastAPI app with bounded per-GPU concurrency
- `start_server.sh`: small launcher for uvicorn
- `test_concurrency.py`: smoke test for queueing behavior
- `README.md`: usage notes

## Start the server

From the repo root:

```bash
bash online_judge/start_server.sh
```

Or directly:

```bash
python -m uvicorn online_judge.app_with_queue:app --host 0.0.0.0 --port 12017
```

## Environment variables

- `PORT`: server port, default `12017`
- `HOST`: bind host, default `0.0.0.0`
- `AVAILABLE_GPUS`: comma-separated GPU ids, default uses all visible GPUs
- `GPU_ALLOCATION_MODE`: `auto` or `manual`, default `auto`
- `MAX_CONCURRENT_EVALS`: concurrent requests per GPU, default `1`
- `MAX_QUEUE_SIZE`: max queued requests per GPU, default `100`
- `RUNTIME_TIMEOUT`: runtime timeout in seconds after a slot is acquired, default `600`

## Endpoints

- `GET /health`: service and GPU status
- `GET /queue/status`: current queue depth and active workers
- `POST /evaluate`: normal KernelBench/SYN evals, plus TBG dispatch via `test_source="TBG"`
- `POST /evaluate_fit_kernel`: FIT / MLSYS remote evals

## Quick test

```bash
python online_judge/test_concurrency.py --url http://127.0.0.1:12017 --requests 4
```
