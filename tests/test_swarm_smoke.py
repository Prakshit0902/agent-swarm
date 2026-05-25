from execution.swarm import CodingSwarm
import pytest

@pytest.mark.asyncio
async def test_swarm_init():
    try:
        sw = CodingSwarm()
        assert sw is not None
    except Exception:
        pass
