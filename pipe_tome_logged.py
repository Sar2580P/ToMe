from pipe_tome import tomePipeline, get_centroid, rescale_noise_cfg, token_merge, retrieve_timesteps
from torchvision import transforms as T
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from utils.ptp_utils import AttentionStore, aggregate_attention, register_self_time
import torch
import numpy as np
from diffusers.image_processor import PipelineImageInput
import pickle 
import os
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.utils import (
    deprecate,
    logging,
)

class CustomTomePipeline(tomePipeline):

    def _entropy_loss(
        self,
        attention_store: AttentionStore,
        indices_to_alter: List[int],
        attention_res: int = 16,
        pose_loss: bool = False,
        mask_save_dir:str= "masks"
    ):
        """Aggregates the attention for each token and computes the max activation value for each token to alter."""
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
        )  # h w 77

        loss = 0

        prompt = self.prompt[0] if isinstance(self.prompt, list) else self.prompt
        last_idx = len(self.tokenizer(prompt)["input_ids"]) - 1

        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text = torch.nn.functional.softmax(
            attention_for_text / 0.5, dim=-1
        )

        # get pos idx and calculate pos loss
        indices = []
        for i in range(len(indices_to_alter)):
            curr_idx = indices_to_alter[i][0][0]
            indices.append(curr_idx)

        indices = [i - 1 for i in indices]
        cross_map = attention_for_text[:, :, indices]  # 32,32 seq_len
        cross_map = (cross_map - cross_map.amin(dim=(0, 1), keepdim=True)) / (
            cross_map.amax(dim=(0, 1), keepdim=True)
            - cross_map.amin(dim=(0, 1), keepdim=True)
        )
        cross_map = cross_map / cross_map.sum(dim=(0, 1), keepdim=True)

        loss = loss - 2 * (cross_map * torch.log(cross_map + 1e-5)).sum()
        if pose_loss:
            idx = 0
            for subject_idx, subject_idx2 in [indices]:
                # Shift indices since we removed the first token
                curr_map = attention_for_text[
                    :, :, [subject_idx, subject_idx2]
                ]  # h w k

                vis_map = curr_map.permute(2, 0, 1)  # k h w
                sub_map, sub_map2 = vis_map[0], vis_map[1]

                sub_map = (sub_map - sub_map.min()) / (sub_map.max() - sub_map.min())
                sub_map2 = (sub_map2 - sub_map2.min()) / (
                    sub_map2.max() - sub_map2.min()
                )

                curr_map = torch.stack([sub_map, sub_map2])  # k h w
                curr_map = curr_map.permute(1, 2, 0)  # h w k
                pair_pos = get_centroid(curr_map) * 32  # (2, 2) k 2

                pos1 = torch.tensor([10.0, 16]).to("cuda")

                pos2 = torch.tensor([25.0, 16]).to("cuda")

                loss = loss + (0.2 * (pair_pos[0] - pos1) ** 2).mean()
                loss = loss + (0.2 * (pair_pos[1] - pos2) ** 2).mean()

                T.ToPILImage()(sub_map.reshape(1, 32, 32)).save("mask_left.png")
                T.ToPILImage()(sub_map2.reshape(1, 32, 32)).save("mask_right.png")
        return loss
    
    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices_to_alter: List[Tuple[int, int]],
        threshold: float,
        text_embeddings: torch.Tensor,
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: int = 32,
        max_refinement_steps: List[int] = [3, 3],
        pose_loss: bool = False,
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code and text embedding according to our loss objective until the given threshold is reached for all tokens.
        """
        entropy_logging = {}
        threshold = threshold / 2 * len(indices_to_alter)
        threshold -= 2
        ratio = t / 1000
        if ratio > 0.9:
            max_refinement_steps = max_refinement_steps[0]
        if ratio <= 0.9:
            max_refinement_steps = max_refinement_steps[1]
        iteration = 0
        entropy_logging['threshold'] = threshold
        entropy_logging['ratio'] = ratio
        entropy_logging['t'] = t
        entropy_logging['pose_loss'] = pose_loss
        loss_vals = []
        
        while True:
            iteration += 1
            torch.cuda.empty_cache()
            latents = latents.clone().detach().requires_grad_(True)
            text_embeddings = text_embeddings.clone().detach().requires_grad_(True)

            noise_pred_text = self.unet(
                latents,
                t,
                encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            loss = self._entropy_loss(
                attention_store, indices_to_alter, attention_res, pose_loss=pose_loss
            )
            loss_vals.append(loss)
            if loss != 0:  # and t/1000 > 0.8:
                latents = self._update_latent(latents, loss, step_size)
                text_embeddings = self._update_text(text_embeddings, loss, step_size)

            if loss < threshold:
                break
            if iteration >= max_refinement_steps:
                print(
                    f"Entropy loss optimization Exceeded max number of iterations ({max_refinement_steps}) "
                )
                break
        entropy_logging['loss'] = loss_vals
        return latents, loss, text_embeddings.detach() , entropy_logging
    
    def opt_token(self, latents: torch.Tensor, t, stoken, prompt_anchor, iter_num=3):
        """
        latents: 128 128 4
        stoken: dim
        prompt_anchor: 77 dim
        """
        stoken.requires_grad_(True)

        latents = latents.clone().detach().unsqueeze(0)
        iteration = 0
        opt_token_logging = {}
        opt_token_logging['t'] = t
        opt_token_logging['iter_num'] = iter_num
        loss_vals = []
        with torch.no_grad():
            noise_pred_anchor = self.unet(
                latents,
                t,
                encoder_hidden_states=prompt_anchor,
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample
        while True:
            iteration += 1
            noise_pred_token = self.unet(
                latents,
                t,
                encoder_hidden_states=stoken.unsqueeze(0).unsqueeze(0),
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred_anchor, noise_pred_token)
            loss_vals.append(loss)
            stoken = self._update_stoken(stoken, loss, 10000)
            if iteration >= iter_num:
                print(
                    f"Semantic binding loss optimization Exceeded max number of iterations ({iter_num}) "
                )
                break
        
        opt_token_logging['loss'] = loss_vals
        with torch.no_grad():
            noise_pred_null = self.unet(
                latents,
                t,
                encoder_hidden_states=self.negative_prompt_embeds,
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            noise_pred = noise_pred_null + self.guidance_scale * (
                noise_pred_null - noise_pred_anchor
            )

            noise_pred = rescale_noise_cfg(
                noise_pred,
                noise_pred_anchor,
                guidance_rescale=self.guidance_rescale,
            )
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            self.scheduler._step_index -= 1
        return stoken, latents[0], opt_token_logging
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        log_dir : str = 'results' , 
        should_perform_logging: bool = False , 
        **kwargs,
    ):

        callback = None
        callback_steps = None

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        attention_store = kwargs.get("attention_store")
        indices_to_alter = kwargs.get("indices_to_alter")
        attention_res = kwargs.get("attention_res")
        run_standard_sd = kwargs.get("run_standard_sd")
        thresholds = kwargs.get("thresholds")
        scale_factor = kwargs.get("scale_factor")
        scale_range = kwargs.get("scale_range")
        smooth_attentions = kwargs.get("smooth_attentions")
        sigma = kwargs.get("sigma")
        kernel_size = kwargs.get("kernel_size")
        prompt_anchor = kwargs.get("prompt_anchor")
        prompt3 = kwargs.get("prompt3")
        prompt_length = kwargs.get("prompt_length")
        token_refinement_steps = kwargs.get("token_refinement_steps")
        attention_refinement_steps = kwargs.get("attention_refinement_steps")
        tome_control_steps = kwargs.get("tome_control_steps")
        eot_replace_step = kwargs.get("eot_replace_step")
        use_pose_loss = kwargs.get("use_pose_loss")

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.prompt = prompt
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        panchors = []
        for panchor in prompt_anchor:
            (
                prompt_anchor_emb,
                _,
                _,
                _,
            ) = self.encode_prompt(
                prompt=panchor,
                prompt_2=panchor,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            panchors.append(prompt_anchor_emb)

        (
            prompt_anchor3,
            _,
            _,
            _,
        ) = self.encode_prompt(
            prompt=prompt3,
            prompt_2=prompt3,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # stoken1, stoken2 = prompt_embeds[0,2], prompt_embeds[0,6]
        # -----------------------------------
        # token merge
        if not run_standard_sd and token_refinement_steps:
            prompt_embeds[0] = token_merge(prompt_embeds[0], indices_to_alter)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.timestep_cond = timestep_cond
        self._num_timesteps = len(timesteps)
        self.timesteps = timesteps

        scale_range = np.linspace(
            scale_range[0], scale_range[1], len(self.scheduler.timesteps)
        )

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            added_cond_kwargs["image_embeds"] = image_embeds

        # added_cond_kwargs2 = {"text_embeds": add_text_embeds[1:], "time_ids": add_time_ids[1:]}

        added_cond_kwargs2 = {
            "text_embeds": torch.zeros_like(add_text_embeds[1:]),
            "time_ids": add_time_ids[1:],
        }

        self.added_cond_kwargs2 = added_cond_kwargs2
        self.negative_prompt_embeds = negative_prompt_embeds
        self.pos = None

        # del self.text_encoder, self.text_encoder_2
        prompt_embeds2 = None
        latent_anchor = None
        semantic_logging, entropy_logging = [], [] 
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                register_self_time(self, None)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_anchor = (
                    torch.cat([latents] * len(panchors))
                    if latent_anchor is None
                    else latent_anchor
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                latent_anchor = self.scheduler.scale_model_input(latent_anchor, t)

                latents_up = (
                    latent_model_input[1:].clone().detach()
                )  # .requires_grad_(True)

                prompt_embeds2 = (
                    prompt_embeds if prompt_embeds2 is None else prompt_embeds2
                )

                with torch.enable_grad():
                    if not run_standard_sd:
                        token_control, attention_control = tome_control_steps
                        # EOT replace
                        if i == eot_replace_step:
                            prompt_embeds2[1, prompt_length + 1 :] = prompt_anchor3[0][
                                prompt_length + 1 :]
                        # semantic binding loss for token refinement
                        if i < token_control:
                            for idx, (panchor_embed, panchor) in enumerate(zip(panchors, prompt_anchor)):
                                stoken = (
                                    prompt_embeds2[1, indices_to_alter[idx][0][0]]
                                    .detach()
                                    .clone()
                                )
                                stoken, latent_anchor[idx] , log = self.opt_token(
                                    latent_anchor[idx],
                                    t,
                                    stoken,
                                    panchor_embed,
                                    token_refinement_steps,
                                )
                                if should_perform_logging:
                                    log['panchor'] = panchor
                                    semantic_logging.append(log)
                                prompt_embeds2[1, indices_to_alter[idx][0][0]] = stoken
                        # entropy loss for attention refinement
                        if i < attention_control:
                            latents_up, loss, prompt_embeds2, log = (
                                self._perform_iterative_refinement_step(
                                    latents=latents_up,
                                    indices_to_alter=indices_to_alter,
                                    threshold=thresholds[i],
                                    text_embeddings=prompt_embeds2,
                                    attention_store=attention_store,
                                    step_size=scale_factor * scale_range[i],
                                    t=t,
                                    attention_res=attention_res,
                                    max_refinement_steps=attention_refinement_steps,
                                    pose_loss=use_pose_loss,
                                )
                            )
                            if should_perform_logging: entropy_logging.append(log)

                            print(f"Iteration {i} | Loss: {loss:0.4f}")

                latent_model_input = (
                    torch.cat([latents_up] * 2)
                    if self.do_classifier_free_guidance
                    else latents_up
                )
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds2,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                        
        # save the logged semantic and entropy
        if should_perform_logging:
            try:

                file_name = f"{log_dir}/inference_logs.pkl"
                inference_logs = {}
                inference_logs['semantic'] = semantic_logging
                inference_logs['entropy'] = entropy_logging
                inference_logs['kwargs'] = kwargs
                with open(file_name, 'wb') as f:
                    pickle.dump(inference_logs, f)
                print(f"Inference logs saved to {file_name}.pkl")
            except Exception as e:
                print(f"Error saving inference logs: {e}")
                    
        
        
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    
