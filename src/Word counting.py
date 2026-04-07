from pathlib import Path
from collections import Counter
from multiprocessing import Pool, current_process
import re
import time
import statistics
import threading
from queue import Queue

# tokenization with regex
WORD_RE = re.compile(r"\b\w+\b") 
DEBUG = False # if there is no need for printing
def tokenize(text):
    # Convert text to lowercase and extract words using regex
    tokens = WORD_RE.findall(text.lower())
    return tokens

# mapping
def count_words_in_file(file_path):
    # Read one file and count words inside it, while timing each step
    worker_name = current_process().name
    if DEBUG:
        print(f"{worker_name} started {file_path.name}")

    read_start = time.perf_counter()
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    read_end = time.perf_counter()

    tokenize_start = time.perf_counter()
    tokens = tokenize(text)
    tokenize_end = time.perf_counter()

    count_start = time.perf_counter()
    local_counts = Counter(tokens)
    count_end = time.perf_counter()

    if DEBUG:
        print(f"{worker_name} finished {file_path.name}")

    timings = {
        "read_time": read_end - read_start,
        "tokenize_time": tokenize_end - tokenize_start,
        "count_time": count_end - count_start,
        "total_file_time": (count_end - read_start),
    }

    return local_counts, timings
# reducing
def merge_counts(partial_counts):
    # Merge a list of Counter objects into one global Counter
    total_counts = Counter()
    for local_counts in partial_counts:
        total_counts.update(local_counts)
    return total_counts

def summarize_times(times):
    # Return descriptive statistics for a list of runtimes
    return (
        statistics.mean(times),
        statistics.stdev(times) if len(times) > 1 else 0.0,
        min(times),
        max(times),
    )

# ---------------------------- Sequential ----------------------------
def sequential_word_count(corpus_dir):
    file_paths = sorted(corpus_dir.glob("*.txt"))
    partial_counts = []
    total_read_time = 0.0
    total_tokenize_time = 0.0
    total_count_time = 0.0

    map_start = time.perf_counter()
    for file_path in file_paths:
        local_counts, file_timings = count_words_in_file(file_path)
        partial_counts.append(local_counts)

        total_read_time += file_timings["read_time"]
        total_tokenize_time += file_timings["tokenize_time"]
        total_count_time += file_timings["count_time"]
    map_end = time.perf_counter()

    merge_start = time.perf_counter()
    total_counts = merge_counts(partial_counts)
    merge_end = time.perf_counter()

    timings = {
        "read_time": total_read_time,
        "tokenize_time": total_tokenize_time,
        "count_time": total_count_time,
        "map_time": map_end - map_start,
        "merge_time": merge_end - merge_start,
        "total_time": (map_end - map_start) + (merge_end - merge_start),
    }

    return total_counts, timings

# ---------------------------- parallelizing(multiprocess) ----------------------------
def parallel_word_count(corpus_dir, nb_workers):
    # Process files in parallel with multiprocessing and collect detailed timings
    file_paths = sorted(corpus_dir.glob("*.txt"))
    map_start = time.perf_counter()
    with Pool(processes=nb_workers) as pool:
        results = pool.map(count_words_in_file, file_paths)
    map_end = time.perf_counter()
    partial_counts = []
    total_read_time = 0.0
    total_tokenize_time = 0.0
    total_count_time = 0.0

    for local_counts, file_timings in results:
        partial_counts.append(local_counts)
        total_read_time += file_timings["read_time"]
        total_tokenize_time += file_timings["tokenize_time"]
        total_count_time += file_timings["count_time"]
    merge_start = time.perf_counter()
    total_counts = merge_counts(partial_counts)
    merge_end = time.perf_counter()
    timings = {
        "read_time": total_read_time,
        "tokenize_time": total_tokenize_time,
        "count_time": total_count_time,
        "map_time": map_end - map_start,
        "merge_time": merge_end - merge_start,
        "total_time": (map_end - map_start) + (merge_end - merge_start),
    }
    return total_counts, timings

# ---------------------------- parallelizing(multithread) ----------------------------

def thread_worker(file_queue, results, lock):
    # Each thread repeatedly takes a file from the queue and processes it
    while True:
        try:
            file_path = file_queue.get_nowait()  # get task 
        except:
            # Queue is empty so no more work
            break
        local_counts, timings = count_words_in_file(file_path)
        # Safely store result (shared structure)
        with lock:
            results.append((local_counts, timings))
        file_queue.task_done()

def thread_word_count(corpus_dir, nb_workers):
    # Prepare file list
    file_paths = sorted(corpus_dir.glob("*.txt"))
    # Create a queue and fill it with files
    file_queue = Queue()
    for path in file_paths:
        file_queue.put(path)
    # Shared results list + lock for thread-safe writes
    results = []
    lock = threading.Lock()
    # Start timing map phase
    map_start = time.perf_counter()
    # Create and start threads
    threads = []
    for i in range(nb_workers):
        t = threading.Thread(target=thread_worker, args=(file_queue, results, lock))
        t.start()
        threads.append(t)
    # Wait for all threads to finish
    for t in threads:
        t.join()
    map_end = time.perf_counter()
    # Collect partial results
    partial_counts = []
    total_read_time = 0.0
    total_tokenize_time = 0.0
    total_count_time = 0.0
    for local_counts, file_timings in results:
        partial_counts.append(local_counts)
        total_read_time += file_timings["read_time"]
        total_tokenize_time += file_timings["tokenize_time"]
        total_count_time += file_timings["count_time"]
    # Merge/reduce phase
    merge_start = time.perf_counter()
    total_counts = merge_counts(partial_counts)
    merge_end = time.perf_counter()
    timings = {
        "read_time": total_read_time,
        "tokenize_time": total_tokenize_time,
        "count_time": total_count_time,
        "map_time": map_end - map_start,
        "merge_time": merge_end - merge_start,
        "total_time": (map_end - map_start) + (merge_end - merge_start),
    }
    return total_counts, timings


def measurements(run_fn, label, nb_runs, corpus_dir, nb_workers=None):
    # Run one implementation several times and report summary statistics
    times = []
    map_times = []
    merge_times = []
    read_times = []
    tokenize_times = []
    count_times = []
    result = None
    for i in range(nb_runs):
        if nb_workers is None:
            result, timings = run_fn(corpus_dir)
        else:
            result, timings = run_fn(corpus_dir, nb_workers)

        times.append(timings["total_time"])
        map_times.append(timings["map_time"])
        merge_times.append(timings["merge_time"])
        read_times.append(timings["read_time"])
        tokenize_times.append(timings["tokenize_time"])
        count_times.append(timings["count_time"])

        print(
            f"Run {i+1}: total={timings['total_time']:.6f}s | "
            f"map={timings['map_time']:.6f}s | "
            f"merge={timings['merge_time']:.6f}s | "
            f"read={timings['read_time']:.6f}s | "
            f"tokenize={timings['tokenize_time']:.6f}s | "
            f"count={timings['count_time']:.6f}s"
        )

    mean_t, std_t, min_t, max_t = summarize_times(times)
    mean_m, std_m, min_m, max_m = summarize_times(map_times)
    mean_merge, std_merge, min_merge, max_merge = summarize_times(merge_times)

    print(
        f"{label} total={mean_t:.4f}s (std:{std_t:.4f}) "
        f"| map={mean_m:.4f}s "
        f"| merge={mean_merge:.4f}s "
        f"| min={min_t:.4f}s max={max_t:.4f}s"
    )

    return result, mean_t
# ---------------------------- Main ----------------------------
def main():
    corpus_dir = Path("data/corpus")

    print("=== Sequential version ===")
    seq_result, seq_avg_time = measurements(
        sequential_word_count,
        "Sequential",
        5,
        corpus_dir,
    )
    print("\nNumber of unique words:", len(seq_result))
    print("\nTop 10 words:")
    for word, count in seq_result.most_common(10):
        print(word, count)

    print("\n=== Parallel with different numbers of workers ===")
    for nb_workers in [1, 2, 4, 8]:
        print(f"\n--- Testing with {nb_workers} worker(s) ---")
        par_result, par_avg_time = measurements(
            parallel_word_count,
            f"Parallel {nb_workers} workers:",
            5,
            corpus_dir,
            nb_workers,
        )
        print("Results are equal:", seq_result == par_result)
        assert seq_result == par_result, f"Mismatch with {nb_workers} workers"
        speedup = seq_avg_time / par_avg_time
        print("Speed-up:", round(speedup, 3), "x")

    print("\n=== Thread-based parallel version (threading) ===")
    for nb_workers in [1, 2, 4, 8]:
        print(f"\n--- Testing threads with {nb_workers} worker(s) ---")
        thr_result, thr_avg_time = measurements(
            thread_word_count,
            f"Threads {nb_workers} workers:",
            5,
            corpus_dir,
            nb_workers,
        )
        print("Results are equal:", seq_result == thr_result)
        assert seq_result == thr_result
        speedup = seq_avg_time / thr_avg_time
        print("Speed-up:", round(speedup, 3), "x")

if __name__ == "__main__":
    main()

