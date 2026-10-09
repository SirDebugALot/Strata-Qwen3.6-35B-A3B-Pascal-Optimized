# Windows Pascal binary release

This archive contains a Windows x64 executable compiled with CUDA Toolkit 12.9 for NVIDIA Pascal `sm_61`, together with the cuBLAS and Visual C++ runtime DLLs it needs. The model is not included.

Requirements:

- Windows 10 or 11
- NVIDIA Pascal `sm_61` GPU, such as a GTX 1050 Ti, GTX 1060, GTX 1070, or GTX 1080
- A recent NVIDIA driver compatible with CUDA 12.9
- An AVX2-capable CPU
- At least 32 GB of system RAM; 48 GB or more is recommended for the tested GGUF
- Python 3.10 or newer for the web/API launcher

Setup and run:

1. Extract the ZIP to a writable directory.
2. Run `SETUP-RUNTIME.bat` once.
3. Download `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` from [Hugging Face](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf).
4. Start the server:

   ```bat
   START-OPTIMIZED-PASCAL.bat "D:\models\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
   ```

5. Open <http://127.0.0.1:8080>.

The CUDA DLLs in this archive are redistributed under NVIDIA's CUDA Toolkit license. The installed NVIDIA display driver supplies `nvcuda.dll` and is still required.
