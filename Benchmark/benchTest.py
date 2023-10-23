from BenchmarkTool.Dataset import Process_Dataset
from BenchmarkTool.Benchmark import Benchmark


# Example usage:
# Currently supported models: facebook/bart-large-cnn and gpt2
MODEL_NAME = "facebook/bart-large-cnn"
# Supported Datasets: cnn_dailymail, bookcorpus, and xsum
DATASET_NAME = "xsum"
NUM_SAMPLES = 100

# Create dataset instance
DATASET = Process_Dataset(DATASET_NAME, num_samples=NUM_SAMPLES)

# Create benchmark instance and run evaluation
benchmark = Benchmark(MODEL_NAME, DATASET, batch_size=5)
benchmark.run_evaluation()
