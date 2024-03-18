import subprocess

def run_sycl_output():
    try:
        result = subprocess.run(["./sycl_output"], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("Error: File 'sycl_output' not found.")

def run_hip_output():
    try:
        result = subprocess.run(["./hip_output"], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("Error: File 'hip_output' not found.")

def run_cuda_output():
    try:
        result = subprocess.run(["./cuda_output"], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("Error: File 'cuda_output' not found.")

if __name__ == "__main__":
    for _ in range(5):
        print("Output from running 'sycl_output':")
        run_sycl_output()
        print("-" * 30)  # Separating each run output for clarity
    
    print()
    
    for _ in range(5):
        print("Output from running 'hip_output':")
        run_hip_output()
        print("-" * 30)  # Separating each run output for clarity
    
    # for _ in range(5):
    #     print("Output from running 'cuda_output':")
    #     run_cuda_output()
    #     print("-" * 30)
