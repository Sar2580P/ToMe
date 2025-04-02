import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from configs.demo_config import RunConfig1, RunConfig2
from pipe_tome import tomePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
from prompt_utils import PromptParser
import spacy
import os

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def read_prompt(path):
    with open(path, "r") as f:
        prompt_ls = f.readlines()

    all_prompt = []

    for idx, prompt in enumerate(prompt_ls):
        prompt = prompt.replace("\n", "")
        all_prompt.append([idx, prompt])
    return all_prompt


def load_model(config, device):

    stable_diffusion_version = "stabilityai/stable-diffusion-xl-base-1.0"

    if hasattr(config, "model_path") and config.model_path is not None:
        stable_diffusion_version = config.model_path
    stable = tomePipeline.from_pretrained(
        stable_diffusion_version,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    ).to(device)
    # stable.enable_xformers_memory_efficient_attention()
    stable.unet.requires_grad_(False)
    stable.vae.requires_grad_(False)
    # stable.enable_model_cpu_offload()

    prompt_parser = PromptParser(stable_diffusion_version)

    return stable, prompt_parser


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {
        idx: stable.tokenizer.decode(t)
        for idx, t in enumerate(stable.tokenizer(prompt)["input_ids"])
        if 0 < idx < len(stable.tokenizer(prompt)["input_ids"]) - 1
    }
    pprint.pprint(token_idx_to_word)
    token_indices = input(
        "Please enter the a comma-separated list indices of the tokens you wish to "
        "alter (e.g., 2,5): "
    )
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(
    prompt: List[str],
    model: tomePipeline,
    controller: AttentionStore,
    token_indices: List[int],
    prompt_anchor: List[str],
    seed: torch.Generator,
    config,
) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(
        prompt=prompt,
        guidance_scale=config.guidance_scale,
        generator=seed,
        num_inference_steps=config.n_inference_steps,
        attention_store=controller,
        indices_to_alter=token_indices,
        prompt_anchor=prompt_anchor,
        attention_res=config.attention_res,
        run_standard_sd=config.run_standard_sd,
        thresholds=config.thresholds,
        scale_factor=config.scale_factor,
        scale_range=config.scale_range,
        prompt3=config.prompt_merged,
        prompt_length=config.prompt_length,
        token_refinement_steps=config.token_refinement_steps,
        attention_refinement_steps=config.attention_refinement_steps,
        tome_control_steps=config.tome_control_steps,
        eot_replace_step=config.eot_replace_step,
        use_pose_loss=config.use_pose_loss,
        negative_prompt="low res, ugly, blurry, artifact, unreal",
    )
    image = outputs.images[0]
    return image


def filter_text(token_indices, prompt_anchor):
    final_idx = []
    final_prompt = []
    for i, idx in enumerate(token_indices):
        if len(idx[1]) == 0:
            continue
        final_idx.append(idx)
        final_prompt.append(prompt_anchor[i])
    return final_idx, final_prompt


def main():
    config = RunConfig2() #edit this to change the config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stable, prompt_parser = load_model(config, device)
    # ------------------parser prompt-------------------------
    if config.use_nlp:
        # import en_core_web_trf

        nlp = None   #en_core_web_trf.load()  # load spacy

        doc = nlp(config.prompt)
        prompt_parser.set_doc(doc)
        token_indices = prompt_parser._get_indices(config.prompt)
        prompt_anchor = prompt_parser._split_prompt(doc)
        token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)
    else:
        token_indices = config.token_indices
        prompt_anchor = config.prompt_anchor
    # ------------------parser prompt-------------------------

    # token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        print(f"Original Prompt: {config.prompt}")
        print(f"Anchor Prompt: {prompt_anchor}")
        print(f"Indices of merged tokens: {token_indices}")
        g = torch.Generator("cuda").manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(
            prompt=config.prompt,
            model=stable,
            controller=controller,
            token_indices=token_indices,
            prompt_anchor=prompt_anchor,
            seed=g,
            config=config,
        )
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(
            prompt_output_path
            / f'{seed}_{"standard" if config.run_standard_sd else "tome"}.png'
        )
        images.append(image)

    joined_image = vis_utils.get_image_grid(images)

    joined_image.save(
        config.output_path
        / f'{config.prompt}_{"standard" if config.run_standard_sd else "tome"}.png'
    )


if __name__ == "__main__":
    path = 'submodules/examples/dataset/color_val.txt'
    s = read_prompt(path)
    print(s)
    print(type(s[0]))
    idx, pr = s[0]
    print(idx, pr)
    # main()
