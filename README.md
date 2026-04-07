# Parallel Word Counting in Python

This project explores the performance of sequential and parallel implementations of a word counting task in Python.

The goal is not only to compute word frequencies, but also to understand how different implementation choices (multiprocessing, threading, task granularity) affect performance.

---

## 🚀 Implementations

The project includes three versions:

Sequential: processes files one by one
Multiprocessing: uses multiple processes (true parallelism)
Threading: uses multiple threads (limited by GIL)

---

## ⚙️ Requirements

- Python 3.8+
- No external dependencies (only standard library)

---

## ▶️ How to Run

Make sure your corpus is inside:
data/corpus/


Then run:

```bash
python word_counting.py
