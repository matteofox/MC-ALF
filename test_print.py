print("Hello from test_print.py")
try:
    import pypolychord
    print("pypolychord imported successfully")
except ImportError:
    print("pypolychord NOT found")
