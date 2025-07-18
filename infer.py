import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
import os
import sa_handler_for_flux_clean as sa_handler
import argparse
from torch.nn import functional as nnf
from qwen_for_subject import Qwen2VL
from seg_birefnet import BiRefNet
from flux_utils import pil_to_latents_flux


def make_inversion_callback(zts, num_inference_steps):

    def callback_on_step_end(pipeline, i: int, t, callback_kwargs):
        latents = callback_kwargs['latents']
        latents = zts[min(i + 1, num_inference_steps-1)].to(latents.device, latents.dtype)
        return {'latents': latents}
    
    return zts[0], callback_on_step_end
        

def get_args():
    parser = argparse.ArgumentParser(description='share attention before specific steps.')

    parser.add_argument('--extend_scale', type=float, default=1.1)
    parser.add_argument('--mu_shift_type', type=str, default='negative')
    parser.add_argument('--num_inference_steps', type=int, default=30, help='The specific share layer value.')
    parser.add_argument('--qwen2_vl_path', type=str, required=True)
    parser.add_argument('--qwen2_5_path', type=str, required=True)
    parser.add_argument('--birefnet_path', type=str, required=True)
    parser.add_argument('--flux_path', type=str, required=True)
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--subject_word', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='./test.jpg', help='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    vital_layers = [0, 1, 2, 17, 18, 25, 28, 53, 54, 56]
    seed = 777
    with torch.no_grad():
        height, width = 512, 512 
        
        qwen2_vl = Qwen2VL(args.qwen2_vl_path, args.qwen2_5_path)
        
        birefnet = BiRefNet(
            model_path = args.birefnet_path,
            target_size_h=height,
            target_size_w=width,
        )
        model_id = args.flux_path
        pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to('cuda')
        
        handler = sa_handler.Handler()
        num_inference_steps = args.num_inference_steps
        weight_extended = args.extend_scale

        output_path = args.output_path
        prompt = args.prompt
        image_path = args.input_image                    
        x0 = Image.open(image_path).resize((height, height))
        
        # get foreground region
        masked_image, mask = birefnet.extract(image_path)
        mask_label = torch.from_numpy(np.array(mask)).to('cuda', dtype=torch.bfloat16)
        mask_label = mask_label.view(1, 1, mask_label.size()[0], mask_label.size()[1])
        resized_mask = nnf.interpolate(mask_label, (height // 16 , width // 16), mode='bilinear')
        mask_label = resized_mask.view(1, 1, (height // 16) * (width // 16), 1)
        indices = (mask_label > 0).nonzero(as_tuple=True)
        dim_neg2_indices = indices[-2]
        mask_ = mask_label / 255

                        
        _, img2img_latents, latent_image_ids = pil_to_latents_flux(x0, pipeline, num_inference_steps, height=height, width=width, seed=seed, shift_type=args.mu_shift_type)
        zT, inversion_callback = make_inversion_callback(img2img_latents, num_inference_steps)

        # get prompts
        subject_word = 'subject' if args.subject_word is None else args.subject_word
        subject_description = qwen2_vl.get_filtered_description(prompt, subject_word)


        handler.register(pipeline, num_inference_steps=num_inference_steps, mode='save', ref_dim_neg2_indices=dim_neg2_indices,vital_layers=vital_layers,)              
        
        prompt = prompt + '. ' + subject_description
            
        # get the attention of input image
        _ = pipeline(
            prompt=prompt,
            latents=zT,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=inversion_callback,
        ).images
        

        handler.register(pipeline, num_inference_steps=num_inference_steps, mode='use', scale=weight_extended, ref_dim_neg2_indices=dim_neg2_indices, vital_layers=vital_layers,)
        image = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        
        handler.clear()
        image.save(output_path)


