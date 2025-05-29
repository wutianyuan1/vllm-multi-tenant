curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "prompt": "A GEMM kernel with coalesced memory access is implemented as follows:",
        "max_tokens": 200,
        "temperature": 0
    }'