"""Report which required packages are installed and their versions."""
import importlib

MODULES = [
    "tensorflow", "keras", "numpy", "pandas", "matplotlib",
    "seaborn", "PIL", "sklearn", "tqdm", "cv2",
]

missing = []
for m in MODULES:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  OK   {m:14s} {ver}")
    except ImportError as e:
        print(f"  MISS {m:14s} ({e.__class__.__name__})")
        missing.append(m)

print()
if missing:
    print(f"Missing: {', '.join(missing)}")
else:
    print("All dependencies present.")
