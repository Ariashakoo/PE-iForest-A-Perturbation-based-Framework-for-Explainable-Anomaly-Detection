import os
import warnings
from scripts.benchmark_speed import run_speed_benchmark
from scripts.benchmark_fidelity import run_fidelity_benchmark

warnings.filterwarnings("ignore")

def main():
    # Define local data path or mount point
    DATA_PATH = './datasets/' 
    
    if not os.path.exists(DATA_PATH):
        print(f"Data path '{DATA_PATH}' not found. Please create the directory and add the required datasets.")
        return

    print("Initiating PE-Framework Benchmarks...")
    
    # 1. Run Speed Benchmark across multiple models
    run_speed_benchmark(DATA_PATH)
    
    # 2. Run Sensitivity/Fidelity Benchmark on CreditCard data
    run_fidelity_benchmark(DATA_PATH)

if __name__ == "__main__":
    main()