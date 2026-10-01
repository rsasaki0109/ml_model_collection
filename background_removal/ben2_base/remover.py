"""BEN2-Base: shared square-input runner (tools/mlmc/matting.py)."""

from tools.mlmc.matting import SquareMatting


class Remover(SquareMatting):
    """As upstream onnx_run.py: no mean/std, alpha min-max rescaled after upsampling."""

    def __init__(self, model, provider="cuda"):
        super().__init__(model, provider, normalize=False, output="prob", minmax=True)
