"""
Multi-Stream FPS Benchmark for Wildlife Detection

Tests inference throughput with multiple concurrent video streams to determine
viable stream counts for deployment.

Reports:
- FPS per stream at different concurrency levels
- Total throughput (frames processed per second)
- Latency statistics (min, max, avg, p95, p99)

Usage:
    python benchmark_fps.py --model best.pt --source videos/test.mp4
    python benchmark_fps.py --model best.pt --source videos/test.mp4 --streams 1 2 4 8
    python benchmark_fps.py --model best.pt --source 0  # Use webcam
"""

import argparse
import json
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Check if CUDA is available
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        GPU_NAME = "N/A (CPU)"
except ImportError:
    CUDA_AVAILABLE = False
    GPU_NAME = "N/A (torch not available)"

from ultralytics import YOLO


class StreamBenchmarkResult:
    """Results from benchmarking a single stream."""
    
    def __init__(self, stream_id: int):
        self.stream_id = stream_id
        self.frames_processed = 0
        self.total_time = 0.0
        self.latencies: List[float] = []  # ms per frame
        self.errors = 0
    
    @property
    def fps(self) -> float:
        if self.total_time == 0:
            return 0
        return self.frames_processed / self.total_time
    
    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0
        return statistics.mean(self.latencies)
    
    @property
    def p95_latency(self) -> float:
        if len(self.latencies) < 20:
            return max(self.latencies) if self.latencies else 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[idx]
    
    @property
    def p99_latency(self) -> float:
        if len(self.latencies) < 100:
            return max(self.latencies) if self.latencies else 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[idx]
    
    def to_dict(self) -> Dict:
        return {
            "stream_id": self.stream_id,
            "frames_processed": self.frames_processed,
            "total_time_seconds": round(self.total_time, 3),
            "fps": round(self.fps, 2),
            "latency_ms": {
                "min": round(min(self.latencies), 2) if self.latencies else 0,
                "max": round(max(self.latencies), 2) if self.latencies else 0,
                "avg": round(self.avg_latency, 2),
                "p95": round(self.p95_latency, 2),
                "p99": round(self.p99_latency, 2),
            },
            "errors": self.errors,
        }


class ConcurrencyBenchmarkResult:
    """Results from benchmarking at a specific concurrency level."""
    
    def __init__(self, num_streams: int):
        self.num_streams = num_streams
        self.stream_results: List[StreamBenchmarkResult] = []
        self.wall_clock_time = 0.0
    
    @property
    def total_frames(self) -> int:
        return sum(r.frames_processed for r in self.stream_results)
    
    @property
    def total_fps(self) -> float:
        """Total throughput across all streams."""
        if self.wall_clock_time == 0:
            return 0
        return self.total_frames / self.wall_clock_time
    
    @property
    def avg_fps_per_stream(self) -> float:
        if not self.stream_results:
            return 0
        return statistics.mean(r.fps for r in self.stream_results)
    
    @property
    def avg_latency(self) -> float:
        all_latencies = []
        for r in self.stream_results:
            all_latencies.extend(r.latencies)
        if not all_latencies:
            return 0
        return statistics.mean(all_latencies)
    
    def to_dict(self) -> Dict:
        return {
            "num_streams": self.num_streams,
            "total_frames": self.total_frames,
            "wall_clock_seconds": round(self.wall_clock_time, 3),
            "total_fps": round(self.total_fps, 2),
            "avg_fps_per_stream": round(self.avg_fps_per_stream, 2),
            "avg_latency_ms": round(self.avg_latency, 2),
            "streams": [r.to_dict() for r in self.stream_results],
        }


def benchmark_single_stream(
    stream_id: int,
    model_path: str,
    source: str,
    num_frames: int,
    conf: float,
    imgsz: int,
) -> StreamBenchmarkResult:
    """
    Benchmark a single stream.
    
    Each stream loads its own model instance to simulate real multi-stream deployment.
    """
    result = StreamBenchmarkResult(stream_id)
    
    # Load model (each stream gets its own instance)
    model = YOLO(model_path)
    
    # Open video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        result.errors = num_frames
        return result
    
    start_time = time.perf_counter()
    
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            # Loop video if we run out of frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                result.errors += 1
                continue
        
        # Time inference
        frame_start = time.perf_counter()
        _ = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        frame_end = time.perf_counter()
        
        latency_ms = (frame_end - frame_start) * 1000
        result.latencies.append(latency_ms)
        result.frames_processed += 1
    
    result.total_time = time.perf_counter() - start_time
    cap.release()
    
    return result


def run_concurrent_benchmark(
    model_path: str,
    source: str,
    num_streams: int,
    frames_per_stream: int,
    conf: float,
    imgsz: int,
) -> ConcurrencyBenchmarkResult:
    """
    Run benchmark with multiple concurrent streams.
    """
    result = ConcurrencyBenchmarkResult(num_streams)
    
    print(f"\n  Running {num_streams} concurrent stream(s)...", end=" ", flush=True)
    
    wall_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_streams) as executor:
        futures = [
            executor.submit(
                benchmark_single_stream,
                stream_id=i,
                model_path=model_path,
                source=source,
                num_frames=frames_per_stream,
                conf=conf,
                imgsz=imgsz,
            )
            for i in range(num_streams)
        ]
        
        for future in as_completed(futures):
            stream_result = future.result()
            result.stream_results.append(stream_result)
    
    result.wall_clock_time = time.perf_counter() - wall_start
    
    print(f"Done. Total: {result.total_fps:.1f} FPS, Per-stream: {result.avg_fps_per_stream:.1f} FPS")
    
    return result


def warmup_model(model_path: str, source: str, imgsz: int):
    """Run a few inference passes to warm up the model."""
    print("Warming up model...", end=" ", flush=True)
    
    model = YOLO(model_path)
    
    # Try to get a frame from the source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            # Run 5 warmup inferences
            for _ in range(5):
                _ = model.predict(frame, conf=0.5, imgsz=imgsz, verbose=False)
        cap.release()
    
    print("Done")


def print_benchmark_report(results: List[ConcurrencyBenchmarkResult], model_path: str, source: str):
    """Print formatted benchmark report."""
    
    print("\n" + "=" * 70)
    print("MULTI-STREAM FPS BENCHMARK REPORT")
    print("=" * 70)
    
    print(f"\n📌 Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Source: {source}")
    print(f"   GPU: {GPU_NAME}")
    print(f"   CUDA Available: {CUDA_AVAILABLE}")
    
    print(f"\n📊 Results:")
    print("-" * 70)
    print(f"{'Streams':>8} | {'Total FPS':>10} | {'FPS/Stream':>12} | {'Avg Latency':>12} | {'P95 Latency':>12}")
    print("-" * 70)
    
    for result in results:
        all_latencies = []
        for sr in result.stream_results:
            all_latencies.extend(sr.latencies)
        
        if all_latencies:
            sorted_lat = sorted(all_latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            p95 = sorted_lat[p95_idx] if p95_idx < len(sorted_lat) else sorted_lat[-1]
        else:
            p95 = 0
        
        print(f"{result.num_streams:>8} | {result.total_fps:>10.1f} | {result.avg_fps_per_stream:>12.1f} | {result.avg_latency:>10.1f}ms | {p95:>10.1f}ms")
    
    print("-" * 70)
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    # Find sweet spot (highest total FPS with acceptable per-stream FPS)
    if results:
        best_total = max(results, key=lambda r: r.total_fps)
        best_per_stream = max(results, key=lambda r: r.avg_fps_per_stream)
        
        # Consider 10 FPS per stream as minimum for smooth processing
        viable = [r for r in results if r.avg_fps_per_stream >= 10]
        if viable:
            recommended = max(viable, key=lambda r: r.num_streams)
            print(f"   - Recommended max streams: {recommended.num_streams} (maintains ≥10 FPS/stream)")
        
        print(f"   - Best total throughput: {best_total.num_streams} streams @ {best_total.total_fps:.0f} FPS total")
        print(f"   - Best per-stream FPS: {best_per_stream.num_streams} stream(s) @ {best_per_stream.avg_fps_per_stream:.0f} FPS each")
        
        # Latency warning
        high_latency = [r for r in results if r.avg_latency > 100]
        if high_latency:
            print(f"   - ⚠️  High latency (>100ms) at: {[r.num_streams for r in high_latency]} streams")


def main():
    parser = argparse.ArgumentParser(description="Multi-stream FPS benchmark")
    
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--source", required=True, 
                        help="Video file path, RTSP URL, or camera index (0)")
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4],
                        help="Number of concurrent streams to test (default: 1 2 4)")
    parser.add_argument("--frames", type=int, default=100,
                        help="Frames to process per stream (default: 100)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (default: 0.35)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--output", type=str, default="benchmark_report.json",
                        help="Output JSON file")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip model warmup")
    
    args = parser.parse_args()
    
    # Validate source exists (handle paths with spaces)
    if not args.source.isdigit():
        source_path = Path(args.source)
        if not source_path.exists() and not Path(args.source).is_file():
            print(f"Error: Source not found: {args.source}")
            print("\nPlease provide a valid video file or camera index.")
            print("Examples:")
            print("  python benchmark_fps.py --model best.pt --source videos/test.mp4")
            print("  python benchmark_fps.py --model best.pt --source 0  # webcam")
            return
    
    print("=" * 70)
    print("MULTI-STREAM FPS BENCHMARK")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Source: {args.source}")
    print(f"Streams to test: {args.streams}")
    print(f"Frames per stream: {args.frames}")
    print(f"Input size: {args.imgsz}")
    print(f"Confidence: {args.conf}")
    print(f"GPU: {GPU_NAME}")
    
    # Warmup
    if not args.no_warmup:
        warmup_model(args.model, args.source, args.imgsz)
    
    # Run benchmarks at each concurrency level
    results = []
    
    for num_streams in sorted(args.streams):
        result = run_concurrent_benchmark(
            model_path=args.model,
            source=args.source,
            num_streams=num_streams,
            frames_per_stream=args.frames,
            conf=args.conf,
            imgsz=args.imgsz,
        )
        results.append(result)
    
    # Print report
    print_benchmark_report(results, args.model, args.source)
    
    # Save results
    output_data = {
        "model": args.model,
        "source": args.source,
        "frames_per_stream": args.frames,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "gpu": GPU_NAME,
        "cuda_available": CUDA_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "results": [r.to_dict() for r in results],
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Report saved to: {args.output}")


if __name__ == "__main__":
    main()
