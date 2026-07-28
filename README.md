# mlx-quant-mcp

**MCP server for [ternary quantization (BitNet b1.58)](https://github.com/8b-is/mlx-quant)**
over [Model Context Protocol](https://modelcontextprotocol.io) `2026-07-28`.

## Tools

- `quantize`
- ` dequantize`
- ` matmul`
- ` maybe_matmul`
- ` mergeq_matmul`
- ` gather_qmm`
- ` info`

## Usage

```bash
pip install mcp
python server.py
```

Requires: MLX-QUANT (pip install mlx).

## The 8b-is MCP Ecosystem

| MCP Server | Purpose |
|------------|---------|
| **honest-irc-mcp** | Quantum-proof messaging + honesty-auth |
| **ayeos-mcp** | Ternary matrix inference (LINOSV seed) |
| **mlx-quant-mcp** | Ternary quantization (BitNet b1.58) |
| **bluesky-mcp** | AT Protocol (24 tools) |

**[axiomquant.org](https://axiomquant.org)** · **[pocoo.vaked.dev](https://pocoo.vaked.dev)**
