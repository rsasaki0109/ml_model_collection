"""SAFMN-x4: shared x4 runner, input reflect-padded to multiples of 8
(the exported graph replaces adaptive max-pooling, see export.py)."""

from tools.mlmc.sr import SuperRes


class Upscaler(SuperRes):
    def __init__(self, model, provider="cuda"):
        super().__init__(model, provider, pad_multiple=8)
