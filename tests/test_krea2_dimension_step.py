import inspect
import sys

import PIL.Image
import pytest

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.krea2.cli import krea2_generate
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.scale_factor import ScaleFactor


class _StopAfterGeneration(Exception):
    pass


class _FakeKrea2:
    captured: dict = {}

    def __init__(self, **kwargs):
        pass

    def generate_image(self, **kwargs):
        _FakeKrea2.captured = kwargs
        raise _StopAfterGeneration


def _run_cli(monkeypatch, *argv):
    monkeypatch.setattr(krea2_generate, "Krea2", _FakeKrea2)
    monkeypatch.setattr(krea2_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "test", *argv])
    with pytest.raises(_StopAfterGeneration):
        krea2_generate.main()
    return _FakeKrea2.captured


@pytest.mark.fast
@pytest.mark.parametrize(
    "width, height, step, expected",
    [
        (1320, 1984, 16, (1312, 1984)),
        (1320, 1984, 8, (1320, 1984)),
        (1315, 1983, 8, (1312, 1976)),
        (1000, 1000, 16, (992, 992)),
        (1000, 1000, 8, (1000, 1000)),
        (1024, 1024, 16, (1024, 1024)),
        (1024, 1024, 8, (1024, 1024)),
    ],
)
def test_config_rounds_down_to_the_dimension_step(width, height, step, expected):
    config = Config(model_config=ModelConfig.krea2(), width=width, height=height, dimension_step=step)

    assert (config.width, config.height) == expected


@pytest.mark.fast
def test_config_keeps_rounding_to_16_unless_a_step_is_given():
    # Every other model builds Config without a step; they must not change.
    config = Config(model_config=ModelConfig.krea2(), width=1320, height=1984)

    assert (config.width, config.height) == (1312, 1984)


@pytest.mark.fast
def test_config_warning_names_the_step_in_use(caplog):
    with caplog.at_level("WARNING"):
        Config(model_config=ModelConfig.krea2(), width=1315, height=1024, dimension_step=8)

    assert "multiples of 8" in caplog.text


@pytest.mark.fast
def test_krea2_generate_image_defaults_to_a_step_of_8():
    assert inspect.signature(Krea2.generate_image).parameters["dimension_step"].default == 8


@pytest.mark.fast
@pytest.mark.parametrize(
    "argv, expected_step",
    [
        ([], 8),
        (["--model", "krea-2-raw"], 8),
        (["--legacy-sizes"], 16),
        (["--model", "krea-2-raw", "--legacy-sizes"], 16),
    ],
)
def test_cli_passes_the_dimension_step_to_generation(monkeypatch, argv, expected_step):
    assert _run_cli(monkeypatch, *argv)["dimension_step"] == expected_step


@pytest.fixture
def source_image(tmp_path):
    path = tmp_path / "source.png"
    PIL.Image.new("RGB", (1000, 600)).save(path)
    return path


@pytest.mark.fast
@pytest.mark.parametrize(
    "scale, kwargs, expected",
    [
        ("1x", {}, (992, 592)),  # every other caller: still the 16 step
        ("1x", {"pixel_step": 16}, (992, 592)),
        ("1x", {"pixel_step": 8}, (1000, 600)),
        ("1.5x", {"pixel_step": 16}, (1488, 896)),
        ("1.5x", {"pixel_step": 8}, (1496, 896)),
    ],
)
def test_dimension_resolver_snaps_scale_factors_to_the_pixel_step(source_image, scale, kwargs, expected):
    resolved = DimensionResolver.resolve(
        width=ScaleFactor.parse(scale),
        height=ScaleFactor.parse(scale),
        reference_image_path=source_image,
        **kwargs,
    )

    assert resolved == expected


@pytest.mark.fast
def test_dimension_resolver_leaves_explicit_sizes_alone_at_any_step():
    assert DimensionResolver.resolve(width=1001, height=999, pixel_step=8) == (1001, 999)
    assert DimensionResolver.resolve(width=1001, height=999, pixel_step=16) == (1001, 999)


@pytest.mark.fast
@pytest.mark.parametrize(
    "extra_argv, expected_size",
    [
        ([], (1000, 600)),
        (["--legacy-sizes"], (992, 592)),
        (["--width", "1.5x", "--height", "1.5x"], (1496, 896)),
        (["--legacy-sizes", "--width", "1.5x", "--height", "1.5x"], (1488, 896)),
    ],
)
def test_cli_auto_and_scaled_sizes_follow_the_dimension_step(monkeypatch, source_image, extra_argv, expected_size):
    captured = _run_cli(monkeypatch, "--image-path", str(source_image), *extra_argv)

    assert (captured["width"], captured["height"]) == expected_size
