from run_demo import read_prompt, load_model, filter_text, run_on_prompt
import torch
from utils.ptp_utils import AttentionStore
from pathlib import Path
import argparse
import subprocess
import os

def get_file_name(idx, seed, prompt):
    idx = str(idx).zfill(6)
    return f"{prompt}_seed-{seed}_{idx}.png"

def get_BLIPvqa_eval(img_folder_path: str):
    """Run the BLIP VQA evaluation script on the provided image folder"""
    cmd = f"python submodules/BLIPvqa_eval/BLIP_vqa.py --out_dir {img_folder_path}"
    subprocess.run(cmd, shell=True)

def generate_images_from_prompts(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stable, prompt_parser = load_model(config, device)
    # ------------------parser prompt-------------------------
    
    for ele in config["prompts"]:
        idx, prompt = ele
        if config["use_nlp"]:
            import en_core_web_trf
            
            nlp = en_core_web_trf.load()  # load spacy
            doc = nlp(prompt)
            prompt_parser.set_doc(doc)
            token_indices = prompt_parser._get_indices(prompt)
            prompt_anchor = prompt_parser._split_prompt(doc)
            token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)
        else:
            raise NotImplementedError("Prompt Anchors not supported for bulk prompt generation, use nlp=True")
        # ------------------parser prompt-------------------------
        
        for seed in config["seeds"]:
            print(f"Seed: {seed}")
            print(f"Original Prompt: {prompt}")
            print(f"Anchor Prompt: {prompt_anchor}")
            print(f"Indices of merged tokens: {token_indices}")
            g = torch.Generator("cuda").manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(
                prompt=prompt,
                model=stable,
                controller=controller,
                token_indices=token_indices,
                prompt_anchor=prompt_anchor,
                seed=g,
                config=config,
            )
            prompt_output_path = config["output_path"] / get_file_name(idx, seed, prompt)
            image.save(
                prompt_output_path
            )

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate images from prompts in a file")
    parser.add_argument("--prompt_file", type=str, default="submodules/examples/dataset/color_val.txt",
                        help="Path to the prompt file")
    parser.add_argument("--save_dir", type=str, default="results/color_val",
                        help="Directory to save results")
    parser.add_argument("--run_eval", action="store_true",
                        help="Run BLIP VQA evaluation after generating images")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 198],
                        help="Seeds for image generation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--no_nlp", action="store_true",
                        help="Disable NLP processing", default=False)
    parser.add_argument("--run_standard_sd", action="store_true",
                        help="Run standard Stable Diffusion", default=False)
    
    args = parser.parse_args()
    
    # Read prompts
    prompts = read_prompt(args.prompt_file)
    
    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Configure the generation
    config = {
        "model_path": args.model_path,
        "prompts": prompts,
        "use_nlp": not args.no_nlp,
        "seeds": args.seeds,
        "output_path": save_dir,
        "run_standard_sd": args.run_standard_sd
    }
    
    if not os.path.isdir(config["output_path"]):
        os.makedirs(config["output_path"], exist_ok=True)
        # Generate images
        generate_images_from_prompts(config)
    
    # Run evaluation if requested
    if args.run_eval:
        print(f"Running BLIP VQA evaluation on {args.save_dir}")
        get_BLIPvqa_eval(args.save_dir)