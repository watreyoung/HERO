#!/bin/bash

# Ensure script errors don't stop other background tasks, but can log
# Create log directory
# mkdir -p llm_logs

echo "Starting parallel execution on 8 GPUs..."

# ====================================================
# GPU 0 & 1: CodeLlama-7b
# ====================================================
echo "Launching CodeLlama-7b (CPP) on GPU 0..."
nohup python llm.py --model_name "CodeLlama-7b" --language "cpp" --gpu_id "0" > llm_logs/codellama_cpp.log 2>&1 &

echo "Launching CodeLlama-7b (Java) on GPU 1..."
nohup python llm.py --model_name "CodeLlama-7b" --language "java" --gpu_id "1" > llm_logs/codellama_java.log 2>&1 &

# Brief pause to prevent system memory (RAM) overflow from simultaneous loading
sleep 100

# ====================================================
# GPU 2 & 3: DeepSeek-Coder-7b
# ====================================================
echo "Launching DeepSeek-Coder-7b (CPP) on GPU 2..."
nohup python llm.py --model_name "DeepSeek-Coder-7b" --language "cpp" --gpu_id "2" > llm_logs/deepseek_cpp.log 2>&1 &

echo "Launching DeepSeek-Coder-7b (Java) on GPU 3..."
nohup python llm.py --model_name "DeepSeek-Coder-7b" --language "java" --gpu_id "3" > llm_logs/deepseek_java.log 2>&1 &

sleep 180

# ====================================================
# GPU 4 & 5: Yi-Coder-9b
# ====================================================
echo "Launching Yi-Coder-9b (CPP) on GPU 4..."
nohup python llm.py --model_name "Yi-Coder-9b" --language "cpp" --gpu_id "4" > llm_logs/yi_cpp.log 2>&1 &

echo "Launching Yi-Coder-9b (Java) on GPU 5..."
nohup python llm.py --model_name "Yi-Coder-9b" --language "java" --gpu_id "5" > llm_logs/yi_java.log 2>&1 &

sleep 180

# # ====================================================
# # GPU 6 & 7: Qwen2.5-Coder-7b
# # ====================================================
echo "Launching Qwen2.5-Coder-7b (CPP) on GPU 6..."
nohup python llm.py --model_name "Qwen2.5-Coder-7b" --language "cpp" --gpu_id "6" > llm_logs/qwen_cpp.log 2>&1 &

echo "Launching Qwen2.5-Coder-7b (Java) on GPU 7..."
nohup python llm.py --model_name "Qwen2.5-Coder-7b" --language "java" --gpu_id "7" > llm_logs/qwen_java.log 2>&1 &

# # ====================================================

echo "All tasks launched in background."
echo "Check the 'llm_logs' folder for progress."
echo "Waiting for all processes to finish..."

# # Wait for all background tasks to complete
# # wait

echo "All experiments completed!"