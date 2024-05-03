#  The MIT License (MIT)
#  Copyright © 2023 Yuma Rao
#  Copyright © 2024 WOMBO
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#
#

from asyncio import Semaphore
from typing import List, Optional, Union, Tuple, Dict, Any, cast

import torch
from PIL import Image
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
    is_compiled_module, StableDiffusionXLControlNetPipeline, is_torch_version,
    ControlNetModel, MultiControlNetModel,
)
from diffusers.utils.torch_utils import randn_tensor

from gpu_pipeline.pipeline import parse_input_parameters
from gpu_pipeline.tensor import load_tensor
from image_generation_protocol.cryptographic_sample import cryptographic_sample
from image_generation_protocol.io_protocol import ImageGenerationInputs


# Credits to Huggingface for the SDXL pipeline code
def __similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    return torch.cosine_similarity(tensor_a.flatten(), tensor_b.flatten(), eps=1e-3, dim=0).item()


@torch.no_grad()
def __validate_internal_cn(
    pipeline: StableDiffusionXLControlNetPipeline,
    index: int,
    frames_tup: Tuple[torch.Tensor, torch.Tensor],
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    image: Image.Image = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
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
    ip_adapter_image=None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    guess_mode: bool = False,
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 1.0,
    original_size: Tuple[int, int] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Tuple[int, int] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    **kwargs,
) -> float:
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders.
        image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
            The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
            specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
            accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
            and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
            `init`, images must be passed as a list such that each element of the list can be correctly batched for
            input to a single ControlNet.
        height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image. Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image. Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 5.0):
            A higher guidance scale value encourages the model to generate images closely linked to the text
            `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in image generation. If not defined, you need to
            pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
            and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
            to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, pooled text embeddings are generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
            weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
            argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
            [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
            The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
            to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
            the corresponding scale as a list.
        guess_mode (`bool`, *optional*, defaults to `False`):
            The ControlNet encoder tries to recognize the content of the input image even if you remove all
            prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
        control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
            The percentage of total steps at which the ControlNet starts applying.
        control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
            The percentage of total steps at which the ControlNet stops applying.
        original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
            `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
            explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
            `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
            `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            For most cases, `target_size` should be set to the desired height and width of the generated image. If
            not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
            section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a specific image resolution. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a target image resolution. It should be as same
            as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeine class.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
            otherwise a `tuple` is returned containing the output images.
    """

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)
    controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

    # align format for control guidance
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt,
        prompt_2,
        image,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
        callback_on_step_end_tensor_inputs,
    )

    pipeline._guidance_scale = guidance_scale
    pipeline._clip_skip = clip_skip
    pipeline._cross_attention_kwargs = cross_attention_kwargs

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device

    global_pool_conditions = (
        controlnet.config.global_pool_conditions
        if isinstance(controlnet, ControlNetModel)
        else controlnet.nets[0].config.global_pool_conditions
    )
    guess_mode = guess_mode or global_pool_conditions

    # 3.1 Encode input prompt
    text_encoder_lora_scale = (
        pipeline.cross_attention_kwargs.get("scale", None) if pipeline.cross_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt,
        prompt_2,
        device,
        num_images_per_prompt,
        pipeline.do_classifier_free_guidance,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
        clip_skip=pipeline.clip_skip,
    )

    # 3.2 Encode ip_adapter_image
    assert ip_adapter_image is None

    # 4. Prepare image
    if isinstance(controlnet, ControlNetModel):
        image = pipeline.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        height, width = image.shape[-2:]
    elif isinstance(controlnet, MultiControlNetModel):
        images = []

        for image_ in image:
            image_ = pipeline.prepare_image(
                image=image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

            images.append(image_)

        image = images
        height, width = image[0].shape[-2:]
    else:
        assert False

    # 5. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    pipeline._num_timesteps = len(timesteps)

    # 6. Prepare latent variables
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6.5 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if pipeline.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = pipeline.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    # 7.1 Create tensor stating which controlnets to keep
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

    # 7.2 Prepare added time ids & embeddings
    if isinstance(image, list):
        original_size = original_size or image[0].shape[-2:]
    else:
        original_size = original_size or image.shape[-2:]
    target_size = target_size or (height, width)

    add_text_embeds = pooled_prompt_embeds
    if pipeline.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipeline._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if pipeline.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    is_unet_compiled = is_compiled_module(pipeline.unet)
    is_controlnet_compiled = is_compiled_module(pipeline.controlnet)
    is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

    latents, expected_next_latents = frames_tup
    t = timesteps[index]

    if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
        torch._inductor.cudagraph_mark_step_begin()
    # expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # controlnet(s) inference
    if guess_mode and pipeline.do_classifier_free_guidance:
        # Infer ControlNet only for the conditional batch.
        control_model_input = latents
        control_model_input = pipeline.scheduler.scale_model_input(control_model_input, t)
        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
        controlnet_added_cond_kwargs = {
            "text_embeds": add_text_embeds.chunk(2)[1],
            "time_ids": add_time_ids.chunk(2)[1],
        }
    else:
        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds
        controlnet_added_cond_kwargs = added_cond_kwargs

    if isinstance(controlnet_keep[i], list):
        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
    else:
        controlnet_cond_scale = controlnet_conditioning_scale
        if isinstance(controlnet_cond_scale, list):
            controlnet_cond_scale = controlnet_cond_scale[0]
        cond_scale = controlnet_cond_scale * controlnet_keep[i]

    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
        control_model_input,
        t,
        encoder_hidden_states=controlnet_prompt_embeds,
        controlnet_cond=image,
        conditioning_scale=cond_scale,
        guess_mode=guess_mode,
        added_cond_kwargs=controlnet_added_cond_kwargs,
        return_dict=False,
    )

    if guess_mode and pipeline.do_classifier_free_guidance:
        # Infered ControlNet only for the conditional batch.
        # To apply the output of ControlNet to both the unconditional and conditional batches,
        # add 0 to the unconditional batch to keep it unchanged.
        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

    # predict the noise residual
    noise_pred = pipeline.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=pipeline.cross_attention_kwargs,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

    # perform guidance
    if pipeline.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    for i in range(index - 1):
        randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )

    # compute the previous noisy sample x_t -> x_t-1
    latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

    return __similarity(cast(torch.Tensor, latents), expected_next_latents)


async def score_similarity(
    gpu_semaphore: Semaphore,
    pipeline: StableDiffusionXLControlNetPipeline,
    frames: bytes,
    inputs: ImageGenerationInputs,
) -> float:
    frames_tensor = load_tensor(frames)

    if frames_tensor.shape[0] != inputs.num_inference_steps:
        return 0.0

    num_random_indices = min(3, inputs.num_inference_steps)

    if frames_tensor.shape[0] - 1 < num_random_indices:
        return 0.0

    random_indices = sorted(
        cryptographic_sample(
            range(frames_tensor.shape[0] - 1),
            k=num_random_indices
        )
    )

    input_kwargs = parse_input_parameters(inputs)
    frames_tensor = frames_tensor.to(pipeline.unet.device, pipeline.unet.dtype)

    async with gpu_semaphore:
        similarities = torch.tensor(
            [
                __validate_internal_cn(
                    pipeline,
                    i + 1,
                    (frames_tensor[i], frames_tensor[i + 1]),
                    **input_kwargs
                )
                for i in random_indices
            ]
        )

    return max(similarities.min().item() * 0.5 + 0.5, 0.0)
