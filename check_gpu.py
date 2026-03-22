import torch

def check_gpu():
    """
    Checks for GPU availability and prints system information.
    """
    print("--- GPU Setup Verification ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)

        print(f"✅ PyTorch has access to the GPU.")
        print(f"Number of available GPUs: {gpu_count}")
        print(f"Current GPU ID: {current_device}")
        print(f"Current GPU Name: {gpu_name}")
    else:
        print("❌ PyTorch does NOT have access to the GPU. Training will be on CPU.")
    print("-----------------------------")

if __name__ == "__main__":
    check_gpu()