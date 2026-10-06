import asyncio
import base64
import json
from pathlib import Path
import runpy
import struct
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class PackedTests(unittest.TestCase):
    def setUp(self):
        self.mx = MagicMock()
        mlx = types.ModuleType('mlx')
        mlx.core = self.mx
        with patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': self.mx}):
            self.module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'server.py'))
        self.mx.dequantize.return_value.shape = (2, 64)
        self.mx.dequantize.return_value.flatten.return_value.tolist.return_value = [0.0] * 128
        self.words = [0x12345678, 0x9abcdef0, 0, 0xffffffff] * 2
        self.encoded = base64.b64encode(struct.pack('<8I', *self.words)).decode()

    def call(self, **kwargs):
        args = dict(codes_b64=self.encoded, scales='[[1.0],[2.0]]', shape='2,64')
        args.update(kwargs)
        return asyncio.run(self.module['dequantize'](**args))

    def test_words_and_matrix_layout_reach_backend(self):
        result = json.loads(self.call())
        self.mx.array.assert_any_call(self.words, dtype=self.mx.uint32)
        self.mx.array.return_value.reshape.assert_called_once_with((2, 4))
        self.mx.array.assert_any_call([[1.0], [2.0]], dtype=self.mx.float32)
        self.assertEqual(result['shape'], [2, 64])

    def test_malformed_inputs_rejected_before_backend(self):
        for kwargs in [dict(codes_b64='!'), dict(codes_b64='AAAA'),
                       dict(shape='0,64'), dict(shape='2,63'), dict(shape='2,64,1'),
                       dict(bits=4), dict(group_size=0), dict(scales='[[1.0]]'),
                       dict(scales='[[NaN],[2.0]]'), dict(scales='[[true],[2.0]]')]:
            with self.subTest(kwargs=kwargs):
                self.mx.reset_mock()
                with self.assertRaises(ValueError):
                    self.call(**kwargs)
                self.mx.dequantize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
