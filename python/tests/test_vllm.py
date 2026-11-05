"""
Unit tests for continuous batching vLLM serving engine.
"""

import pytest
import asyncio
from python.model_pipeline.serve_vllm import ContinuousBatchingEngine


@pytest.mark.asyncio
async def test_continuous_batching_stream():
    engine = ContinuousBatchingEngine(model_name="meta-llama/Meta-Llama-3-8B", tensor_parallel_size=2)
    
    tokens = []
    async for token in engine.generate_stream("What is distributed GPU inference?", max_tokens=10):
        tokens.append(token)
        
    assert len(tokens) == 10
    full_text = "".join(tokens)
    assert len(full_text) > 0
    assert "GPU" in full_text or "inference" in full_text

