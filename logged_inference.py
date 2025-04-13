import pprint
from typing import List
from tqdm import tqdm
import pyrallis
import torch
from PIL import Image
from omegaconf import OmegaConf
from pipe_tome_logged import CustomTomePipeline
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
    stable = CustomTomePipeline.from_pretrained(
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
    model: CustomTomePipeline,
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
        should_perform_logging=config.should_perform_logging, 
        log_dir=config.output_path, 
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


def read_yaml(file_path):
    conf = OmegaConf.load(file_path)
    config = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))
    return config

def run(tome_config):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stable, prompt_parser = load_model(tome_config, device)
    # ------------------parser prompt-------------------------
    if tome_config.use_nlp:
        # import en_core_web_trf

        nlp = None   #en_core_web_trf.load()  # load spacy

        doc = nlp(tome_config.prompt)
        prompt_parser.set_doc(doc)
        token_indices = prompt_parser._get_indices(tome_config.prompt)
        prompt_anchor = prompt_parser._split_prompt(doc)
        token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)
    else:
        token_indices = tome_config.token_indices
        prompt_anchor = tome_config.prompt_anchor
    # ------------------parser prompt-------------------------

    # token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    images = []
    for seed in tome_config.seeds:
        print(f"Seed: {seed}")
        print(f"Original Prompt: {tome_config.prompt}")
        print(f"Anchor Prompt: {prompt_anchor}")
        print(f"Indices of merged tokens: {token_indices}")
        save_dir = os.path.join(tome_config.output_path , 
                                "with_tome" if not tome_config.run_standard_sd else "without_tome" , 
                                tome_config.prompt)
        os.makedirs(save_dir, exist_ok=True)
        tome_config.output_path = save_dir
        g = torch.Generator("cuda").manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(
            prompt=tome_config.prompt,
            model=stable,
            controller=controller,
            token_indices=token_indices,
            prompt_anchor=prompt_anchor,
            seed=g,
            config=tome_config,
        )

        
        image.save(f"{save_dir}/generated_image.png")
        images.append(image)

    joined_image = vis_utils.get_image_grid(images)

    joined_image.save(f"{save_dir}/seeds-[{tome_config.seeds}] generation.png")


def main(config_path):
    config = read_yaml(config_path)
    tome_config = config['tome_params']
    assert len(config['prompts_merged'])==len(config['prompts']), "Mismatch, len(prompts_merged)!=len(prompts)... check config"
    
    for idx, (prompt, prompt_merged) in tqdm(enumerate(zip(config['prompts'], config['prompts_merged'])), 
                                             desc=f"Running ToMe on {len(config['prompts'])} prompts..."):
        tome_config['prompt'], tome_config['prompt_merged'], tome_config['prompt_length'] = prompt, prompt_merged, len(prompt)
        
        run(tome_config=tome_config)

if __name__ == "__main__":
    main('configs/config.yaml')
