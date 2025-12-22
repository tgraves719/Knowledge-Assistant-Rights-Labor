"""
Performance Benchmark Script for Karl API Server.

Tests:
- Response latency (p50, p95, p99)
- Throughput (requests/second)
- Concurrent request handling
- Endpoint-specific performance
"""

import asyncio
import time
import statistics
from typing import List, Dict
import httpx
from dataclasses import dataclass
import json


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    endpoint: str
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    requests_per_second: float
    total_time_seconds: float


class APIBenchmark:
    """Benchmark runner for Karl API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/api/health")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    async def single_request(self, endpoint: str, method: str = "POST", payload: dict = None) -> tuple[bool, float]:
        """Make a single request and return (success, latency_ms)."""
        start = time.time()
        try:
            if method == "GET":
                response = await self.client.get(f"{self.base_url}{endpoint}")
            else:
                response = await self.client.post(f"{self.base_url}{endpoint}", json=payload)
            
            latency_ms = (time.time() - start) * 1000
            success = 200 <= response.status_code < 300
            return success, latency_ms
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            print(f"Request failed: {e}")
            return False, latency_ms
    
    async def run_benchmark(
        self,
        endpoint: str,
        method: str = "POST",
        payload: dict = None,
        num_requests: int = 100,
        concurrency: int = 10
    ) -> BenchmarkResult:
        """Run benchmark with specified parameters."""
        print(f"\nBenchmarking {endpoint}...")
        print(f"  Requests: {num_requests}, Concurrency: {concurrency}")
        
        latencies: List[float] = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request():
            nonlocal successful, failed
            async with semaphore:
                success, latency = await self.single_request(endpoint, method, payload)
                latencies.append(latency)
                if success:
                    successful += 1
                else:
                    failed += 1
        
        # Create all tasks
        tasks = [make_request() for _ in range(num_requests)]
        
        # Run all requests
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if latencies:
            latencies.sort()
            avg_latency = statistics.mean(latencies)
            p50 = latencies[int(len(latencies) * 0.50)]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            min_latency = min(latencies)
            max_latency = max(latencies)
            rps = num_requests / total_time
        else:
            avg_latency = p50 = p95 = p99 = min_latency = max_latency = 0
            rps = 0
        
        return BenchmarkResult(
            endpoint=endpoint,
            total_requests=num_requests,
            successful=successful,
            failed=failed,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            requests_per_second=rps,
            total_time_seconds=total_time
        )
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results in a formatted table."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\n{result.endpoint}")
            print(f"  Total Requests: {result.total_requests}")
            print(f"  Successful: {result.successful} | Failed: {result.failed}")
            print(f"  Success Rate: {(result.successful/result.total_requests*100):.1f}%")
            print(f"\n  Latency (ms):")
            print(f"    Average: {result.avg_latency_ms:.2f}")
            print(f"    P50:     {result.p50_latency_ms:.2f}")
            print(f"    P95:     {result.p95_latency_ms:.2f}")
            print(f"    P99:     {result.p99_latency_ms:.2f}")
            print(f"    Min:     {result.min_latency_ms:.2f}")
            print(f"    Max:     {result.max_latency_ms:.2f}")
            print(f"\n  Throughput: {result.requests_per_second:.2f} req/s")
            print(f"  Total Time: {result.total_time_seconds:.2f}s")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def main():
    """Run comprehensive benchmarks."""
    benchmark = APIBenchmark()
    
    print("="*80)
    print("KARL API PERFORMANCE BENCHMARKS")
    print("="*80)
    
    # Check server health
    print("\n[1] Checking server health...")
    if not await benchmark.health_check():
        print("ERROR: Server is not responding. Make sure it's running on http://127.0.0.1:8000")
        await benchmark.close()
        return
    
    print("[OK] Server is healthy")
    
    # Test queries
    test_queries = [
        {
            "question": "What is my overtime rate?",
            "user_classification": "all_purpose_clerk",
            "hours_worked": 40,
            "months_employed": 12
        },
        {
            "question": "What holidays do we get off?",
            "user_classification": None,
            "hours_worked": 0,
            "months_employed": 0
        },
        {
            "question": "How does vacation scheduling work?",
            "user_classification": "courtesy_clerk",
            "hours_worked": 0,
            "months_employed": 6
        }
    ]
    
    results = []
    
    # Benchmark health endpoint
    print("\n[2] Benchmarking /api/health...")
    health_result = await benchmark.run_benchmark(
        endpoint="/api/health",
        method="GET",
        num_requests=200,
        concurrency=20
    )
    results.append(health_result)
    
    # Benchmark query endpoint with different queries
    print("\n[3] Benchmarking /api/query...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: {query['question'][:50]}...")
        query_result = await benchmark.run_benchmark(
            endpoint="/api/query",
            method="POST",
            payload=query,
            num_requests=50,
            concurrency=5
        )
        results.append(query_result)
    
    # Benchmark wage endpoint
    print("\n[4] Benchmarking /api/wage...")
    wage_result = await benchmark.run_benchmark(
        endpoint="/api/wage",
        method="POST",
        payload={
            "classification": "courtesy_clerk",
            "hours_worked": 0,
            "months_employed": 24
        },
        num_requests=200,
        concurrency=20
    )
    results.append(wage_result)
    
    # Print all results
    benchmark.print_results(results)
    
    # Save results to file
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump([{
            "endpoint": r.endpoint,
            "total_requests": r.total_requests,
            "successful": r.successful,
            "failed": r.failed,
            "avg_latency_ms": r.avg_latency_ms,
            "p50_latency_ms": r.p50_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "p99_latency_ms": r.p99_latency_ms,
            "min_latency_ms": r.min_latency_ms,
            "max_latency_ms": r.max_latency_ms,
            "requests_per_second": r.requests_per_second,
            "total_time_seconds": r.total_time_seconds
        } for r in results], f, indent=2)
    
    print(f"\n[OK] Results saved to {output_file}")
    
    await benchmark.close()


if __name__ == "__main__":
    asyncio.run(main())



