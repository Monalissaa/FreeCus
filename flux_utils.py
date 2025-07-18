import torch
import numpy as np
from diffusers.utils.torch_utils import randn_tensor


def retrieve_latents(
    encoder_output: torch.Tensor, generator = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    

def _encode_vae_image(sd_pipe, image: torch.Tensor, generator: torch.Generator):
    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(sd_pipe.vae.encode(image[i : i + 1]), generator=generator[i])
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(sd_pipe.vae.encode(image), generator=generator)

    image_latents = (image_latents - sd_pipe.vae.config.shift_factor) * sd_pipe.vae.config.scaling_factor

    return image_latents


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2, height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pil_to_latents_flux(pil_image, sd_pipe, num_inference_steps, height=1024, width=1024, seed=0, shift_type='linear'):
    # import pdb; pdb.set_trace()
    image = sd_pipe.image_processor.preprocess(pil_image, height=height, width=width)
    dtype = torch.bfloat16
    image = image.to(device='cuda', dtype=dtype)
    image_latents = _encode_vae_image(sd_pipe=sd_pipe, image=image, generator=None)
    image_latents = torch.cat([image_latents], dim=0)
    # image_latents = image_latents * sd_pipe.vae.config.scaling_factor
    batch_size = len(image_latents)
    num_channels_latents = sd_pipe.transformer.config.in_channels // 4
    # height = 2 * (int(height) // sd_pipe.vae_scale_factor)
    # width = 2 * (int(width) // sd_pipe.vae_scale_factor)
    height = 2 * (int(height) // (sd_pipe.vae_scale_factor * 2))
    width = 2 * (int(width) // (sd_pipe.vae_scale_factor * 2))
    shape = (batch_size, num_channels_latents, height, width)
    device = 'cuda'
    generator=torch.Generator("cpu").manual_seed(seed)
    noise = randn_tensor(shape, generator=generator, device=image_latents.device, dtype=dtype)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    # image_seq_len = (height // 2) * (width // 2)
    image_seq_len = (int(height) // sd_pipe.vae_scale_factor // 2) * (int(width) // sd_pipe.vae_scale_factor // 2)
    img2img_latents = []
    
    
    mu = calculate_shift(
            image_seq_len,
            sd_pipe.scheduler.config.base_image_seq_len,
            sd_pipe.scheduler.config.max_image_seq_len,
            sd_pipe.scheduler.config.base_shift,
            sd_pipe.scheduler.config.max_shift,
        )
    if shift_type=='negative':
        sd_pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=-mu)
        timesteps = sd_pipe.scheduler.timesteps
    elif shift_type=='half_negative':
        sd_pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=(-0.5*mu))
        timesteps = sd_pipe.scheduler.timesteps
    elif shift_type=='twice_negative':
        sd_pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=(-2*mu))
        timesteps = sd_pipe.scheduler.timesteps
    elif shift_type=='normal':
        sd_pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = sd_pipe.scheduler.timesteps
    elif shift_type=='linear':
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * sd_pipe.scheduler.config.num_train_timesteps
        timesteps = timesteps.floor().to(torch.float32)
        
        tmp_timesteps = np.linspace(1, sd_pipe.scheduler.config.num_train_timesteps, sd_pipe.scheduler.config.num_train_timesteps, dtype=np.float32)[::-1].copy()
        tmp_timesteps = torch.from_numpy(tmp_timesteps).to(dtype=torch.float32)
        tmp_sigmas = tmp_timesteps / sd_pipe.scheduler.config.num_train_timesteps
        sd_pipe.scheduler.timesteps = tmp_sigmas * sd_pipe.scheduler.config.num_train_timesteps
        sd_pipe.scheduler._step_index = None
        sd_pipe.scheduler._begin_index = None

        sd_pipe.scheduler.sigmas = tmp_sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        sd_pipe.scheduler.sigma_min = sd_pipe.scheduler.sigmas[-1].item()
        sd_pipe.scheduler.sigma_max = sd_pipe.scheduler.sigmas[0].item()
    
    for timestep in timesteps:
        timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        img2img_latent = sd_pipe.scheduler.scale_noise(image_latents, timestep, noise)
        img2img_latent = sd_pipe._pack_latents(img2img_latent, batch_size, num_channels_latents, height, width)
        img2img_latents.append(img2img_latent)
        
    image_latents = sd_pipe._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
    
    latent_image_ids = prepare_latent_image_ids(batch_size, height, width, device, dtype)
    
    return image_latents, img2img_latents, latent_image_ids
