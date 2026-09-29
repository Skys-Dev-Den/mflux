import warnings

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.krea2.latent_creator import Krea2LatentCreator
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# Krea-2 turbo defaults (reference: 8 steps, CFG 1.0, er_sde; sigmas use the official
# dynamic exponential shift, base/max 0.5/1.15 over image seq len 256..6400). The 8-step
# count lives in ui_defaults.MODEL_INFERENCE_STEPS under this model's registry key; the
# parser applies it, so main() sees an already-resolved args.steps.
DEFAULT_MODEL = "krea-2"

# Same architecture as Turbo, so this CLI also runs the undistilled base checkpoint. Raw
# gets its own --steps default (28) from the same registry table, and needs CFG: at 1.0 it
# renders murky, so its guidance default follows Krea's recommended 3.5.
FAMILY_MODELS = ("krea-2-raw",)
DEFAULT_GUIDANCE = {"krea/Krea-2-Turbo": 1.0, "krea/Krea-2-Raw": 3.5}


CONDITIONAL_OPTIONS = {
    "--negative-prompt": {
        "condition": "guidance other than 1.0",
        "reason": "the encoder builds the unconditional branch only when guidance != 1.0, so at the "
        "distilled default of 1.0 the negative prompt is never encoded.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Krea-2 based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    return parser


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load the model (--model accepts only the krea-2 family aliases; anything else errors
    # so a foreign name is never silently run as Krea-2-Turbo)
    model_config = ConfigResolution.resolve_restricted(
        args.model,
        DEFAULT_MODEL,
        model_path=args.model_path,
        extra_keys=FAMILY_MODELS,
        base_model=args.base_model,
    )
    model = Krea2(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
    )

    # 2. Register callbacks (stepwise image output, memory stats, battery saver)
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Krea2LatentCreator,
    )

    try:
        guidance = args.guidance if args.guidance is not None else DEFAULT_GUIDANCE[model_config.model_name]
        if guidance == 1.0 and CommandLineParser._option_was_provided("--negative-prompt"):
            # The declared condition, checked once the default has resolved: the encoder
            # only builds the unconditional branch when guidance != 1.0.
            warnings.warn(
                "--negative-prompt is ignored at guidance 1.0; " + CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
                stacklevel=2,
            )
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                num_inference_steps=args.steps,
                height=height,
                width=width,
                guidance=guidance,
                scheduler=args.scheduler,
                negative_prompt=args.negative_prompt,
                image_path=args.image_path,
                image_strength=args.image_strength,
                pid_decode=args.pid_decode,
                pid_degrade_sigma=args.pid_degrade_sigma,
            )
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
