import torch
from unittest.mock import MagicMock, patch
from logic.src.utils.functions.lexsort import _torch_lexsort_cuda, torch_lexsort

class TestLexSort:
    """Class for lexsort tests."""

    def test_torch_lexsort_cpu(self):
        """Test fallback to numpy lexsort on CPU."""
        k1 = torch.tensor([1, 1, 2, 2])
        k2 = torch.tensor([4, 3, 2, 1])
        keys = (k1, k2)
        idx = torch_lexsort(keys)
        assert idx.dtype == torch.int64
        expected = torch.tensor([3, 2, 1, 0])
        assert torch.equal(idx, expected)

    def test_torch_lexsort_cuda_path(self):
        """Test the pure-pytorch implementation used for CUDA."""
        k1 = torch.tensor([1, 1, 2, 2])
        k2 = torch.tensor([4, 3, 2, 1])
        keys = (k1, k2)
        idx = _torch_lexsort_cuda(keys)
        assert idx.shape == k1.shape
        assert idx.dtype == torch.int64

    def test_torch_lexsort_cuda_call(self):
        """Test that torch_lexsort calls the cuda version if is_cuda is set."""
        k1 = MagicMock()
        k1.is_cuda = True
        keys = (k1,)
        with patch("logic.src.utils.functions.lexsort._torch_lexsort_cuda") as mock_cuda:
            torch_lexsort(keys)
            mock_cuda.assert_called_once()
