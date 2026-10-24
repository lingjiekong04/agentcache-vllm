#!/usr/bin/env bash
# Source after materialization, and explicitly select the existing interpreter.
if [[ -z "${AGENTCACHE_PYTHON:-}" || ! -x "$AGENTCACHE_PYTHON" ]]; then
  echo 'Set AGENTCACHE_PYTHON to an existing compatible Linux Python 3.11 first.' >&2
  return 2
fi
agentcache_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AGENTCACHE_PROJECT="$agentcache_repo/build/vllm-0.8.3"
if [[ ! -f "$AGENTCACHE_PROJECT/materialization.json" ]]; then
  echo 'Materialize the pinned source first; no automatic install or download.' >&2
  return 2
fi
export PYTHONPATH="$AGENTCACHE_PROJECT"
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${AGENTCACHE_HF_HOME:-$agentcache_repo/build/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$agentcache_repo/build/cache/vllm"
export TRITON_CACHE_DIR="$agentcache_repo/build/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$agentcache_repo/build/cache/torchinductor"
