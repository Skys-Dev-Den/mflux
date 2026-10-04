import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear


def _bf16_linear(input_dims: int = 64, output_dims: int = 64) -> nn.Linear:
    linear = nn.Linear(input_dims, output_dims, bias=False)
    linear.weight = linear.weight.astype(mx.bfloat16)
    return linear


def _lora(linear: nn.Linear) -> LoRALinear:
    return LoRALinear.from_linear(linear, r=8)


def _lokr(linear: nn.Linear, dora: bool = False) -> LoKrLinear:
    return LoKrLinear.from_linear(
        linear,
        lokr_w1=mx.random.normal((4, 4)),
        lokr_w2=mx.random.normal((16, 16)),
        dora_scale=mx.ones((64,)) if dora else None,
    )


@pytest.mark.parametrize(
    "make_layer",
    [
        pytest.param(_lora, id="lora"),
        pytest.param(_lokr, id="lokr"),
        pytest.param(lambda linear: _lokr(linear, dora=True), id="lokr-dora"),
        pytest.param(lambda linear: FusedLoRALinear(linear, [_lora(linear), _lokr(linear)]), id="fused"),
        pytest.param(
            lambda linear: FusedLoRALinear(linear, [_lora(linear), _lokr(linear, dora=True)]),
            id="fused-dora",
        ),
    ],
)
def test_adapter_output_keeps_activation_dtype(make_layer):
    layer = make_layer(_bf16_linear())
    assert layer(mx.zeros((1, 4, 64), mx.bfloat16)).dtype == mx.bfloat16


def test_adapter_master_weights_stay_float32():
    layer = _lora(_bf16_linear())
    layer(mx.zeros((1, 4, 64), mx.bfloat16))
    assert layer.lora_A.dtype == mx.float32
    assert layer.lora_B.dtype == mx.float32


def test_adapter_gradients_stay_float32():
    layer = _lora(_bf16_linear())
    layer.linear.freeze()
    x = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)

    def loss_fn():
        return layer(x).astype(mx.float32).sum()

    grads = nn.value_and_grad(layer, loss_fn)()[1]
    assert grads["lora_A"].dtype == mx.float32
    assert grads["lora_B"].dtype == mx.float32


def test_float32_activations_are_unchanged():
    mx.random.seed(0)
    linear = nn.Linear(64, 64, bias=False)
    layer = _lora(linear)
    x = mx.random.normal((1, 4, 64))
    expected = linear(x) + layer.scale * mx.matmul(mx.matmul(x, layer.lora_A), layer.lora_B)
    out = layer(x)
    assert out.dtype == mx.float32
    assert mx.array_equal(out, expected)
