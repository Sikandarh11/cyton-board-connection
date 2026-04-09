"""Session runner for sequential chunk history + final subject average result.

This script is backend-only and frontend-friendly:
- Start one session
- Emit chunk results in order (chunk 1, chunk 2, ...)
- Stop session
- Save one EDF for the full session
- Save one session JSON object containing chunk list + final subject stage
"""

import argparse
import json
import time

from main import SessionPipeline


def run_session(config_path: str, duration_sec: float = None) -> dict:
    pipeline = SessionPipeline(config_path)
    start_info = pipeline.start_session()
    print(f"[avg_results] session started: {start_info['session_id']}")

    start_ts = time.time()

    try:
        while True:
            emitted = pipeline.poll_once()
            for item in emitted:
                print(json.dumps(item, ensure_ascii=True))

            if duration_sec is not None and (time.time() - start_ts) >= duration_sec:
                print(f"[avg_results] duration reached: {duration_sec}s")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[avg_results] interrupted by user")

    summary = pipeline.stop_session()

    print("[avg_results] session stopped")
    print(f"[avg_results] total_chunks: {summary.get('total_chunks', 0)}")
    print(f"[avg_results] final_subject_stage: {summary.get('final_subject_stage')}")
    print(f"[avg_results] edf_path: {summary.get('edf_path')}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run session pipeline with averaged final subject stage")
    parser.add_argument(
        "--config",
        default="1. real-time-pipeline\\config_realtime.json",
        help="Path to config_realtime.json",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Auto-stop session after N seconds (optional)",
    )

    args = parser.parse_args()
    run_session(args.config, duration_sec=args.duration_sec)
