import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class LagrangeSieve {

    private static final long[][] phiTable = new long[7][30030];
    private static final Primorial[] primorialTable = new Primorial[7];
    private static final int CACHE_SIZE = 1 << 22;
    private static final int CACHE_MASK = CACHE_SIZE - 1;
    private static final long[] phiCacheKeys = new long[CACHE_SIZE];
    private static final long[] phiCacheValues = new long[CACHE_SIZE];
    private static final long HASH_MULT = 0x9E3779B97F4A7C15L;
    private static final int CORES = Runtime.getRuntime().availableProcessors();
    @SuppressWarnings("unchecked")
    private static final Future<long[]>[] FUTURES = new Future[CORES];
    private static final ExecutorService executor = Executors.newFixedThreadPool(CORES, r -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        t.setName("lp-sieve-worker");
        return t;
    });

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: java -jar lp.jar <n> [-a <runs>] [-p]");
            System.out.println("Example: java -jar lp.jar 1000000 -a 5 -p");
            return;
        }

        long n = -1;
        int runs = 1;
        boolean printPrime = false;

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-a") && i + 1 < args.length) {
                runs = Integer.parseInt(args[i + 1]);
                i++;
            } else if (args[i].equals("-p")) {
                printPrime = true;
            } else {
                try {
                    n = Long.parseLong(args[i]);
                } catch (NumberFormatException e) {
                    System.err.println("Invalid number format: " + args[i]);
                    return;
                }
            }
        }

        if (n < 1) {
            System.err.println("Please provide a valid n (>= 1)");
            return;
        }

        long internalN = n - 1;
        long[] times = new long[runs];
        long result = 0;

        for (int i = 0; i < runs; i++) {
            clearCaches();
            long start = System.nanoTime();
            result = prime(internalN);
            long end = System.nanoTime();
            times[i] = end - start;
        }

        if (runs == 1) {
            System.out.printf("Time taken: %.3f ms\n", times[0] / 1_000_000.0);
        } else {
            long sum = 0, min = Long.MAX_VALUE, max = Long.MIN_VALUE;
            for (long t : times) {
                sum += t;
                if (t < min) min = t;
                if (t > max) max = t;
            }
            double avg = (double) sum / runs;
            double varSum = 0;
            for (long t : times) varSum += (t - avg) * (t - avg);
            double stddev = Math.sqrt(varSum / runs);

            System.out.printf("Runs: %d\n", runs);
            System.out.printf("Average Time: %.3f ms\n", avg / 1_000_000.0);
            System.out.printf("Min Time:     %.3f ms\n", min / 1_000_000.0);
            System.out.printf("Max Time:     %.3f ms\n", max / 1_000_000.0);
            System.out.printf("Std Dev:      %.3f ms\n", stddev / 1_000_000.0);
        }

        if (printPrime) {
            System.out.println(n + "th prime: " + result);
        }
    }

    private static void clearCaches() {
        for (int i = 0; i < 7; i++) {
            Arrays.fill(phiTable[i], 0);
            primorialTable[i] = null;
        }
        Arrays.fill(phiCacheKeys, 0);
        Arrays.fill(phiCacheValues, 0);
    }

    public static long prime(long n) {
        long[] smallPrimes = new long[7];
        long[] buf0 = new long[6000];
        long[] buf1 = new long[6000];

        boolean smallN = n <= 6;
        WheelInit precompute = wheelInit(buf0, buf1, smallPrimes, 6);

        if (smallN) return smallPrimes[(int) n];

        long nRem = n - 6;

        long sieveDiv = 262144 / precompute.info.period;
        long sieveAlign = sieveDiv * precompute.info.period;

        precompute.wheelCandidatesPerSegment = sieveDiv * precompute.info.size;

        long sieveMax = bound(n);
        long numPrimes = sqrtIsh(sieveMax);
        long primesAllocInfos = Math.max(numPrimes, sieveAlign);

        long[] primeValues = new long[(int) primesAllocInfos];
        int[] primeWheelIndices = new int[(int) primesAllocInfos];

        long[] sieveSegm = new long[(int) ((sieveAlign + 63) >>> 6)];
        long primesLen = sieveInit(primeValues, primeWheelIndices, sieveSegm, sieveAlign, precompute);
        long sieveCollectBase = sieveAlign;

        int[] nextWheelIndexTable = new int[30030];
        int curWheelIdx = 0;
        long[] wheel = precompute.wheel;
        int wheelSize = precompute.info.size;
        for (int m = 0; m < 30030; m++) {
            while (curWheelIdx < wheelSize && wheel[curWheelIdx] < m) curWheelIdx++;
            nextWheelIndexTable[m] = (curWheelIdx < wheelSize) ? curWheelIdx : 0;
        }

        while (true) {
            if (primesLen > nRem) return primeValues[(int) nRem];

            if (primesLen < numPrimes) {
                sieveSegm(primeValues, primeWheelIndices, primesLen, sieveSegm, sieveAlign, sieveCollectBase, precompute);
                primesLen = collectPrimes(primeValues, primeWheelIndices, primesLen, numPrimes, sieveSegm, sieveAlign, sieveCollectBase, precompute);
                sieveCollectBase += sieveAlign;
                continue;
            }

            long sieveSkip = lowerBound(n, sieveAlign);
            long numSkipped = pi(sieveSkip - 1, smallPrimes, primeValues, primesLen);
            long count = n - numSkipped;
            long sieveBase = sieveSkip;

            int segmentsPerTask = 32;
            int numTasks = CORES;

            while (true) {
                for (int t = 0; t < numTasks; t++) {
                    final long taskBase = sieveBase + (long) t * segmentsPerTask * sieveAlign;
                    final long pLen = primesLen;
                    final long sAlign = sieveAlign;
                    final WheelInit prec = precompute;
                    final long[] pVals = primeValues;
                    final int[] nwit = nextWheelIndexTable;

                    FUTURES[t] = executor.submit(() -> {
                        long[] localSieve = new long[(int) ((sAlign + 63) >>> 6)];
                        int[] localPWI = new int[(int) pLen];
                        primeInfoFastforward(pVals, localPWI, pLen, taskBase, nwit);

                        long[] counts = new long[segmentsPerTask];
                        long currentBase = taskBase;
                        for (int seg = 0; seg < segmentsPerTask; seg++) {
                            sieveSegm(pVals, localPWI, pLen, localSieve, sAlign, currentBase, prec);
                            counts[seg] = countPrimesInSegment(localSieve, sAlign, prec);
                            currentBase += sAlign;
                        }
                        return counts;
                    });
                }

                boolean foundInBatch = false;
                long targetSegBase = -1;

                for (int t = 0; t < numTasks; t++) {
                    try {
                        long[] counts = FUTURES[t].get();
                        if (!foundInBatch) {
                            for (int seg = 0; seg < segmentsPerTask; seg++) {
                                long segCount = counts[seg];
                                if (count <= segCount) {
                                    targetSegBase = sieveBase + (long) ((long) t * segmentsPerTask + seg) * sieveAlign;
                                    foundInBatch = true;
                                    for (int u = t + 1; u < numTasks; u++) FUTURES[u].cancel(false);
                                    break;
                                }
                                count -= segCount;
                            }
                        }
                    } catch (Exception ignored) {
                    }
                }

                if (foundInBatch) {
                    long[] localSieve = new long[(int) ((sieveAlign + 63) >>> 6)];
                    int[] localPWI = new int[(int) primesLen];
                    primeInfoFastforward(primeValues, localPWI, primesLen, targetSegBase, nextWheelIndexTable);
                    sieveSegm(primeValues, localPWI, primesLen, localSieve, sieveAlign, targetSegBase, precompute);
                    return -findPrime(localSieve, sieveAlign, count, targetSegBase, precompute);
                }

                sieveBase += (long) numTasks * segmentsPerTask * sieveAlign;
            }
        }
    }

    private static WheelInit wheelInit(long[] buf0, long[] buf1, long[] primes, long iteration) {
        buf0[0] = 1;
        buf1[0] = 1;
        buf0[1] = 2;

        long[] wheel = buf0;
        long[] wheelSwap = buf1;
        WheelInfo info = new WheelInfo(1, 1);

        int primeIdx = 0;
        for (long iter = iteration; iter >= 0; iter--) {
            long p = wheel[1];
            primes[primeIdx++] = p;
            if (iter == 0) break;
            info = wheelExtend(wheelSwap, wheel, info);
            long[] tmp = wheel;
            wheel = wheelSwap;
            wheelSwap = tmp;
        }

        int diffIdx = 0;
        long wheelPrev = 1;
        while (diffIdx < info.size) {
            long wheelNext = wheel[diffIdx + 1];
            wheelSwap[diffIdx] = wheelNext - wheelPrev;
            wheelPrev = wheelNext;
            diffIdx++;
        }

        return new WheelInit(wheel, wheelSwap, info);
    }

    private static WheelInfo wheelExtend(long[] newBuf, long[] wheelBuf, WheelInfo info) {
        long wheelPeriod = info.period;
        int wheelSize = info.size;
        long prime = wheelBuf[1];

        long newPeriod = wheelPeriod * prime;
        int newSize = (int) (wheelSize * (prime - 1));

        boolean wheelHasNext = wheelSize > 1;
        int wheelMainIdx = wheelHasNext ? 1 : 0;
        long wheelMainRoll = wheelHasNext ? 0 : wheelPeriod;
        long elimMain = prime * prime;

        int newIdx = 0, wIdx = wheelMainIdx, elimIdx = 0;
        long wRoll = wheelMainRoll, elim = elimMain;

        while (newIdx < newSize) {
            if (wIdx == wheelSize) {
                wRoll += wheelPeriod;
                wIdx = 0;
            }
            long v = wheelBuf[wIdx + 1] + wRoll;
            int wNext = wIdx + 1;
            if (v < elim) {
                newBuf[newIdx + 1] = v;
                newIdx++;
                wIdx = wNext;
            } else {
                elim = wheelBuf[elimIdx + 2] * prime;
                elimIdx++;
                wIdx = wNext;
            }
        }
        return new WheelInfo(newPeriod, newSize);
    }

    private static long sieveInit(long[] primeValues, int[] primeWheelIndices,
                                  long[] sieve, long sieveAlign, WheelInit init) {
        Arrays.fill(sieve, 0L);
        sieve[0] |= (1L << 1);

        long[] wheel = init.wheel;
        long[] delta = init.delta;
        int wheelSize = init.info.size;

        long prime = wheel[1];
        int primeDeltaIdx = 1;
        int primeIdx = 0;

        while (prime < sieveAlign) {
            long c = prime * prime;
            int dIdx = primeDeltaIdx;
            while (c < sieveAlign) {
                sieve[(int) (c >>> 6)] |= (1L << (c & 0x3f));
                c += delta[dIdx] * prime;
                if (++dIdx == wheelSize) dIdx = 0;
            }

            primeValues[primeIdx] = prime;
            primeWheelIndices[primeIdx] = dIdx;
            primeIdx++;

            boolean isComposite = true;
            while (isComposite) {
                prime += delta[primeDeltaIdx];
                if (++primeDeltaIdx == wheelSize) primeDeltaIdx = 0;
                if (prime >= sieveAlign) break;
                isComposite = (sieve[(int) (prime >>> 6)] & (1L << (prime & 0x3f))) != 0;
            }
        }
        return primeIdx;
    }

    private static void sieveSegm(long[] primeValues, int[] primeWheelIndices, long primesLen,
                                  long[] sieveSegm, long sieveAlign, long base, WheelInit init) {
        Arrays.fill(sieveSegm, 0, (int) ((sieveAlign + 63) >>> 6), 0L);

        int wheelSize = init.info.size;
        long[] delta = init.delta;
        long[] wheel = init.wheel;
        long wheelPeriod = init.info.period;

        for (int pIdx = 0; pIdx < primesLen; pIdx++) {
            long prime = primeValues[pIdx];
            int dIdx = primeWheelIndices[pIdx];

            long pWheel = wheel[dIdx];
            long baseRoundUp = (base + prime - 1) / prime;
            long diff = baseRoundUp - pWheel;
            long quotM = (diff + wheelPeriod - 1) / wheelPeriod;
            long idx = (quotM * wheelPeriod + pWheel) * prime - base;

            while (idx < sieveAlign) {
                sieveSegm[(int) (idx >>> 6)] |= (1L << (idx & 0x3f));
                idx += delta[dIdx] * prime;
                if (++dIdx == wheelSize) dIdx = 0;
            }
            primeWheelIndices[pIdx] = dIdx;
        }
    }

    private static void primeInfoFastforward(long[] primeValues, int[] primeWheelIndices,
                                             long primesLen, long sieveBase,
                                             int[] nextWheelIndexTable) {
        for (int i = 0; i < primesLen; i++) {
            long prime = primeValues[i];
            long k = (sieveBase + prime - 1) / prime;
            primeWheelIndices[i] = nextWheelIndexTable[(int) (k % 30030)];
        }
    }

    private static long collectPrimes(long[] primeValues, int[] primeWheelIndices, long primesLen,
                                      long numPrimes, long[] sieveSegm, long sieveAlign,
                                      long base, WheelInit init) {
        int dIdx = 0;
        long sieveIdx = 1;
        long pIdx = primesLen;
        int wheelSize = init.info.size;
        long[] delta = init.delta;

        while (sieveIdx < sieveAlign) {
            if ((sieveSegm[(int) (sieveIdx >>> 6)] & (1L << (sieveIdx & 0x3f))) == 0) {
                primeValues[(int) pIdx] = base + sieveIdx;
                primeWheelIndices[(int) pIdx] = dIdx;
                if (++pIdx == numPrimes) return pIdx;
            }
            sieveIdx += delta[dIdx];
            if (++dIdx == wheelSize) dIdx = 0;
        }
        return pIdx;
    }

    private static long countPrimesInSegment(long[] sieveSegm, long sieveAlign, WheelInit init) {
        int words = (int) ((sieveAlign + 63) >>> 6);
        long composites = 0;
        for (int w = 0; w < words; w++) composites += Long.bitCount(sieveSegm[w]);
        return init.wheelCandidatesPerSegment - composites;
    }

    private static long findPrime(long[] sieveSegm, long sieveAlign, long count,
                                  long base, WheelInit init) {
        int dIdx = 0;
        long sieveIdx = 1;
        long remaining = count;
        int wheelSize = init.info.size;
        long[] delta = init.delta;

        while (sieveIdx < sieveAlign) {
            if ((sieveSegm[(int) (sieveIdx >>> 6)] & (1L << (sieveIdx & 0x3f))) == 0) {
                if (remaining == 0) return -(base + sieveIdx);
                remaining--;
            }
            sieveIdx += delta[dIdx];
            if (++dIdx == wheelSize) dIdx = 0;
        }
        return remaining;
    }

    private static long pi(long x, long[] smallPrimes, long[] primeValues, long primesLen) {
        if (x <= 1) return 0;
        long primesIdxReal = primesLen + 6;
        long prime = loadPrime(smallPrimes, primeValues, primesIdxReal - 1);

        long y = prime, piY = primesIdxReal;
        while (true) {
            if (x == y) return piY;
            long y2 = y * y;
            if (y2 > x) return (phi(x, piY, smallPrimes, primeValues) + piY) - 1;
            piY = (phi(y2, piY, smallPrimes, primeValues) + piY) - 1;
            y = y2;
        }
    }

    private static long phi(long x, long a, long[] smallPrimes, long[] primeValues) {
        if (a < 7) {
            Primorial ps = primorial(a, smallPrimes, primeValues);
            long xDiv = x / ps.primorial;
            int xMod = (int) (x % ps.primorial);
            long cached = phiTable[(int) a][xMod];
            if (cached == 0) {
                cached = phiCompute(xMod, a, smallPrimes, primeValues);
                phiTable[(int) a][xMod] = cached;
            }
            return cached + (xDiv * ps.totient);
        } else if (a < 18) {
            long key = (a << 48) | x;
            long h = key * HASH_MULT;
            int slot = (int) ((h ^ (h >>> 32)) & CACHE_MASK);
            if (phiCacheKeys[slot] == key) return phiCacheValues[slot];
            long result = phiCompute(x, a, smallPrimes, primeValues);
            phiCacheKeys[slot] = key;
            phiCacheValues[slot] = result;
            return result;
        } else {
            return phiCompute(x, a, smallPrimes, primeValues);
        }
    }

    private static Primorial primorial(long a, long[] smallPrimes, long[] primeValues) {
        Primorial cached = primorialTable[(int) a];
        if (cached != null && cached.primorial != 0) return cached;
        if (a == 0) {
            primorialTable[0] = new Primorial(1, 1);
            return primorialTable[0];
        }
        Primorial prev = primorial(a - 1, smallPrimes, primeValues);
        long prime = loadPrime(smallPrimes, primeValues, a - 1);
        Primorial result = new Primorial(prime * prev.primorial, (prime - 1) * prev.totient);
        primorialTable[(int) a] = result;
        return result;
    }

    private static long phiCompute(long x, long a, long[] smallPrimes, long[] primeValues) {
        long phi = x;
        for (long k = 0; k < a; k++) {
            long pk = loadPrime(smallPrimes, primeValues, k);
            long xDiv = x / pk;
            if (xDiv == 0) return phi;
            long phiRec = phi(xDiv, k, smallPrimes, primeValues);
            if (phiRec == 1) return phi - 1 - countPrimesInEntry(x, k, a, smallPrimes, primeValues);
            phi -= phiRec;
        }
        return phi;
    }

    private static long countPrimesInEntry(long x, long k, long a,
                                           long[] smallPrimes, long[] primeValues) {
        long low = k + 1, high = a - 1, ans = k;
        while (low <= high) {
            long mid = (low + high) >>> 1;
            if (loadPrime(smallPrimes, primeValues, mid) <= x) {
                ans = mid;
                low = mid + 1;
            } else high = mid - 1;
        }
        return ans - k;
    }

    private static long loadPrime(long[] smallPrimes, long[] primeValues, long idx) {
        return (idx <= 6) ? smallPrimes[(int) idx] : primeValues[(int) (idx - 6)];
    }

    private static long bound(long n) {
        if (n < 2) return 4;
        long logN = logIsh(n);
        long logLogN = logIsh(logN);
        return n * (logN + logLogN);
    }

    private static long logIsh(long n) {
        return 64 - Long.numberOfLeadingZeros(n);
    }

    private static long sqrtIsh(long n) {
        int lz = Long.numberOfLeadingZeros(n);
        long blenCeil = 65 - lz;
        return 1L << (blenCeil >>> 1);
    }

    private static long lowerBound(long n, long segmSize) {
        if (n == 0) return 0;
        double nP1 = (double) (n + 1);
        double logN = Math.log(nP1);
        double logLogN = Math.log(logN);
        return ((long) ((nP1 * (logN + logLogN - 1.5)) / segmSize)) * segmSize;
    }

    static class WheelInfo {
        long period;
        int size;

        public WheelInfo(long period, int size) {
            this.period = period;
            this.size = size;
        }
    }

    static class WheelInit {
        long[] wheel;
        long[] delta;
        WheelInfo info;

        long wheelCandidatesPerSegment;

        public WheelInit(long[] wheel, long[] delta, WheelInfo info) {
            this.wheel = wheel;
            this.delta = delta;
            this.info = info;
        }
    }

    static class Primorial {
        long primorial;
        long totient;

        public Primorial(long primorial, long totient) {
            this.primorial = primorial;
            this.totient = totient;
        }
    }
}