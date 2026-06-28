# Installation

## Requirements

| Dependency | Minimum version |
|---|---|
| Python | 3.10 |
| NumPy | 1.24 |
| pandas | 1.5 |

---

## Option 1: Clone and use directly (recommended)

```bash
git clone https://github.com/aryaoutoftouch/architect.git
cd architect
pip install -r requirements.txt
```

Then place `architect.py` on your Python path, or run your scripts from the repository root:

```python
import architect as ar
```

---

## Option 2: Add to an existing project

Copy `architect.py` into your project directory. Ensure `numpy` and `pandas` are installed:

```bash
pip install numpy pandas
```

---

## Verifying the installation

Run a minimal smoke test:

```python
import architect as ar

asset = ar.create(n=100, k=2, random_state=0).generate()
assert asset.series.shape == (100, 4)
print("OK")
```

If this prints `OK` without error, the library is working correctly.

---

## Optional dependencies

| Package | Purpose |
|---|---|
| `pyarrow` or `fastparquet` | Required by `asset.to_parquet()` |
| `matplotlib` | Useful for plotting generated series (not included in library) |

Install as needed:

```bash
pip install pyarrow matplotlib
```

---

## Notes

- Architect has no compiled extensions; no build step is required.
- A PyPI release is planned.
