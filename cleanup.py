import os

files_to_remove = [
    'test_imports.py', 
    'run_tests.py', 
    'test_direct.py', 
    'run_test.bat', 
    'launch_tests.py', 
    'runner.py', 
    'run_verify.py', 
    'full_verification.py', 
    'run_final.py', 
    'final_report.py',
]

for f in files_to_remove:
    if os.path.isfile(f):
        os.remove(f)
        print(f"Removed: {f}")

print("\nCleanup complete!")
