from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax, to_dense_batch
from torch_scatter import scatter, segment_csr


class Aggregation(torch.nn.Module):
    r"""An abstract base class for implementing custom aggregations.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or edge features
          :math:`(|\mathcal{E}|, F_{in})`,
          index vector :math:`(|\mathcal{V}|)` or :math:`(|\mathcal{E}|)`,
        - **output:** graph features :math:`(|\mathcal{G}|, F_{out})` or node
          features :math:`(|\mathcal{V}|, F_{out})`
    """

    # @abstractmethod
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The source tensor.
            index (torch.LongTensor, optional): The indices of elements for
                applying the aggregation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            ptr (torch.LongTensor, optional): If given, computes the
                aggregation based on sorted inputs in CSR representation.
                One of :obj:`index` or :obj:`ptr` must be defined.
                (default: :obj:`None`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim` after aggregation. (default: :obj:`None`)
            dim (int, optional): The dimension in which to aggregate.
                (default: :obj:`-2`)
        """
        pass

    def reset_parameters(self):
        pass

    def __call__(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ) -> Tensor:

        if dim >= x.dim() or dim < -x.dim():
            raise ValueError(
                f"Encountered invalid dimension '{dim}' of "
                f"source tensor with {x.dim()} dimensions"
            )

        if index is None and ptr is None:
            index = x.new_zeros(x.size(dim), dtype=torch.long)

        if ptr is not None:
            if dim_size is None:
                dim_size = ptr.numel() - 1
            elif dim_size != ptr.numel() - 1:
                raise ValueError(
                    f"Encountered invalid 'dim_size' (got "
                    f"'{dim_size}' but expected "
                    f"'{ptr.numel() - 1}')"
                )

        if index is not None:
            if dim_size is None:
                dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
            elif index.numel() > 0 and dim_size <= int(index.max()):
                raise ValueError(
                    f"Encountered invalid 'dim_size' (got "
                    f"'{dim_size}' but expected "
                    f">= '{int(index.max()) + 1}')"
                )

        return super().__call__(x, index, ptr, dim_size, dim, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    # Assertions ##############################################################

    def assert_index_present(self, index: Optional[Tensor]):
        # TODO Currently, not all aggregators support `ptr`. This assert helps
        # to ensure that we require `index` to be passed to the computation:
        if index is None:
            raise NotImplementedError("Aggregation requires 'index' to be specified")

    def assert_sorted_index(self, index: Optional[Tensor]):
        if index is not None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError(
                "Can not perform aggregation since the 'index' " "tensor is not sorted"
            )

    def assert_two_dimensional_input(self, x: Tensor, dim: int):
        if x.dim() != 2:
            raise ValueError(
                f"Aggregation requires two-dimensional inputs " f"(got '{x.dim()}')"
            )

        if dim not in [-2, 0]:
            raise ValueError(
                f"Aggregation needs to perform aggregation in "
                f"first dimension (got '{dim}')"
            )

    # Helper methods ##########################################################

    def reduce(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        reduce: str = "add",
    ) -> Tensor:

        if ptr is not None:
            ptr = expand_left(ptr, dim, dims=x.dim())
            return segment_csr(x, ptr, reduce=reduce)

        assert index is not None
        return scatter(x, index, dim=dim, dim_size=dim_size, reduce=reduce)

    def to_dense_batch(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        fill_value: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:

        # TODO Currently, `to_dense_batch` can only operate on `index`:
        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)

        return to_dense_batch(x, index, batch_size=dim_size, fill_value=fill_value)


###############################################################################


def expand_left(ptr: Tensor, dim: int, dims: int) -> Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        ptr = ptr.unsqueeze(0)
    return ptr


class AttentionalAggregation(Aggregation):
    r"""The soft attention aggregation layer from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPs.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """

    def __init__(self, gate_nn: torch.nn.Module, nn: Optional[torch.nn.Module] = None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:

        self.assert_two_dimensional_input(x, dim)
        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        gate = softmax(gate, index, ptr, dim_size, dim)
        return self.reduce(gate * x, index, ptr, dim_size, dim)

    def masked_forward(
        self,
        x: Tensor,
        mask: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:

        self.assert_two_dimensional_input(x, dim)
        assert index[mask].numel() != 0
        masked_x = x[mask]
        masked_index = index[mask]
        gate = self.gate_nn(x).view(-1, 1)
        masked_x = self.nn(masked_x) if self.nn is not None else masked_x
        masked_gate = softmax(gate[mask], masked_index, ptr, dim_size, dim)
        return self.reduce(masked_gate * masked_x, masked_index, ptr, dim_size, dim)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gate_nn={self.gate_nn}, " f"nn={self.nn})"


class MaskedAttentionalAggregation(Aggregation):
    def __init__(self, gate_nn: torch.nn.Module, nn: Optional[torch.nn.Module] = None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:

        self.assert_two_dimensional_input(x, dim)
        assert index[mask].numel() != 0
        masked_x = x[mask]
        masked_index = index[mask]
        gate = self.gate_nn(x).view(-1, 1)
        masked_x = self.nn(masked_x) if self.nn is not None else masked_x
        masked_gate = softmax(gate[mask], masked_index, ptr, dim_size, dim)
        return self.reduce(masked_gate * masked_x, masked_index, ptr, dim_size, dim)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gate_nn={self.gate_nn}, " f"nn={self.nn})"
