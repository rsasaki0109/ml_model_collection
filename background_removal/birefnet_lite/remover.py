"""BiRefNet-lite: shared square-input runner (tools/mlmc/matting.py)."""

from tools.mlmc.matting import SquareMatting


class Remover(SquareMatting):
    def __init__(self, model, provider="cuda"):
        super().__init__(model, provider, normalize=True, output="logits")
