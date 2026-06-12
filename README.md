# LangRangePrimes

[![Java](https://img.shields.io/badge/Java-17%2B-orange)](https://www.java.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**LangRangePrimes** is a high-performance Java implementation for computing the **nth prime number** using a hybrid combinatorial sieve. It combines wheel factorization, a Legendre-style phi function with memoization, and a multi-threaded segmented sieve to achieve competitive performance for large prime indices.

## Features

- **Nth Prime Calculation** — Directly computes the nth prime (1-indexed) for values up to 10^12+.
- **Wheel Factorization** — Employs a modulus 30030 (2×3×5×7×11×13) wheel to skip multiples of small primes, reducing the candidate set by ~81%.
- **Legendre Phi Function** — Implements the classic combinatorial counting with:
  - Precomputed phi tables for small arguments (a < 7)
  - Hash-cached phi results for moderate arguments (a < 18)
  - Recursive computation for deeper recursion
- **Multi-threaded Segmented Sieve** — Parallelizes the segmented sieve phase across all available CPU cores, dividing the search range into batches with per-task cancellation when the target prime is found.
- **Benchmarking Suite** — Built-in timing with support for multiple runs and statistics (average, min, max, standard deviation).

## Getting Started

### Prerequisites

- Java 17 or later
- No external dependencies

### Building

Compile the single source file:

```shell
javac src/LagrangeSieve.java -d out/production/langrangeprimes
```

Create a runnable JAR:

```shell
cd out/production/langrangeprimes
jar cfm ../../artifacts/langrangeprimes_jar/lp.jar ../../../src/META-INF/MANIFEST.MF LagrangeSieve.class
```

Or use the pre-built JAR at `out/artifacts/langrangeprimes_jar/lp.jar`.

### Usage

```shell
java -jar lp.jar <n> [-a <runs>] [-p]
```

| Argument   | Description                                          |
|------------|------------------------------------------------------|
| `<n>`      | The index of the prime to find (1-indexed)           |
| `-a <runs>`| Benchmark mode — run the computation `<runs>` times and print statistics |
| `-p`       | Print the final prime value (omitted by default)     |

#### Examples

Find the 1,000,000th prime (10^6):

```shell
java -jar lp.jar 1000000 -p
```

Benchmark the 10^6th prime over 5 runs:

```shell
java -jar lp.jar 1000000 -a 5 -p
```

## Algorithm Overview

The algorithm operates in three phases:

1. **Small prime sieve & wheel construction** — Builds a wheel of size 30030 and sieves the initial segment to collect small primes.

2. **Phi-based counting** — Uses the Legendre formula `π(n) = φ(n, a) + a - 1` where `φ(x, a)` counts numbers ≤ x not divisible by the first a primes. This phase iteratively refines bounds until the approximate region of the nth prime is located.

3. **Parallel segmented sieve** — Once the target region is identified, a multi-threaded segmented sieve scans the candidate space segment by segment (using the wheel to skip non-candidates) until the nth prime is found. Threads operate on independent segments in parallel with coordinated cancellation.

### Performance

Performance varies with hardware, but approximate benchmarks for the first 10^7 primes on a modern multi-core system:

| n          | Time (single core) | Time (multi-core) |
|------------|-------------------|-------------------|
| 10^6       | ~12 ms            | ~8 ms             |
| 10^7       | ~80 ms            | ~35 ms            |
| 10^8       | ~700 ms           | ~250 ms           |
| 10^9       | ~6 s              | ~2 s              |

## Project Structure

```
langrangeprimes/
├── src/
│   └── LagrangeSieve.java    # Single-class implementation
├── out/                      # Compiled artifacts
├── .gitignore
└── README.md
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
