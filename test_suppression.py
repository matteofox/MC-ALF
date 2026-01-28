
import os
import sys
import contextlib

def write_to_fd1():
    # Simulate C-level write to stdout
    os.write(1, b"Direct write to FD 1\n")

def write_to_stdout():
    # Regular python print
    print("Python print to stdout")

print("--- Testing redirect_stdout (Expected to FAIL for FD 1) ---")
try:
    with open('/dev/null', 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            write_to_stdout()
            write_to_fd1() 
except Exception as e:
    print(f"Error: {e}")
print("---------------------------------------------------------")

@contextlib.contextmanager
def suppress_output():
    # Open /dev/null
    with open(os.devnull, "w") as devnull:
        # Save old stdout/stderr
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            # Flush existing python buffers
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Redirect
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            
            yield
        finally:
            # Restore
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            # Close saved fds
            os.close(old_stdout)
            os.close(old_stderr)

print("\n--- Testing os.dup2 suppression (Expected to SUCCEED) ---")
print("You should NOT see 'Direct write...' below:")
print("BEGIN HIDDEN SECTION")
with suppress_output():
    write_to_stdout()
    write_to_fd1()
print("END HIDDEN SECTION")
print("---------------------------------------------------------")
