export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda
export CPATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/math_libs/12.9/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/math_libs/12.9/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/math_libs/12.9/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION

# Missing Danish subsets
euroeval --model "common-pile/comma-v0.1-2t" --model "PleIAs/Pleias-350m-Preview" --model "PleIAs/Pleias-1.2b-Preview" --language da --language et --language cs --evaluate-test-split --verbose
euroeval --model "common-pile/comma-v0.1-2t" --language da --language et --language cs --evaluate-test-split --verbose
# euroeval --model "PleIAs/Pleias-3b-Preview" --language da --language et --language cs --evaluate-test-split --verbose

euroeval --model LumiOpen/Viking-7B@100B \
    --model LumiOpen/Viking-7B@300B \
    --model LumiOpen/Viking-7B@500B \
    --model LumiOpen/Viking-7B@700B \
    --model LumiOpen/Viking-7B@900B \
    --language da --verbose