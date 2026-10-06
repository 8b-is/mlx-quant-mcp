"""Real MCP transport tests with MLX isolated; no tensor/GPU validation."""
import asyncio
import inspect
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = """
import sys, types, runpy
mlx = types.ModuleType('mlx')
mlx.core = types.ModuleType('mlx.core')
sys.modules['mlx'] = mlx
sys.modules['mlx.core'] = mlx.core
runpy.run_path(sys.argv[1], run_name='__main__')
"""


class StartupTests(unittest.TestCase):
    def test_console_entry_is_synchronous_and_runs_stdio(self):
        mlx = types.ModuleType('mlx')
        mlx.core = types.ModuleType('mlx.core')
        with patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': mlx.core}):
            module = runpy.run_path(str(ROOT / 'server.py'))
        self.assertFalse(inspect.iscoroutinefunction(module['main']))
        with patch.object(module['server'], 'run') as run:
            module['main']()
            run.assert_called_once_with(transport='stdio')

    def test_real_stdio_initialization_and_tool_discovery(self):
        async def check():
            params = StdioServerParameters(command=sys.executable,
                args=['-c', BOOTSTRAP, str(ROOT / 'server.py')])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    result = await session.initialize()
                    self.assertEqual(result.server_info.name, 'mlx-quant-mcp')
                    tools = await session.list_tools()
                    self.assertEqual({tool.name for tool in tools.tools},
                        {'quantize', 'dequantize', 'matmul', 'maybe_matmul', 'mergeq_matmul', 'info'})
        asyncio.run(asyncio.wait_for(check(), timeout=15))


if __name__ == '__main__':
    unittest.main()
