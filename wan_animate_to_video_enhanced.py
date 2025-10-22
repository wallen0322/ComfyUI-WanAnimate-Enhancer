"""
Enhanced WanAnimateToVideo with multi-dimensional control
Provides independent control over motion, expression, pose, and background
"""

import torch
import types
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
from typing import Tuple, Dict, Any


class WanAnimateToVideoEnhanced:
    """
    Enhanced WanAnimateToVideo node
    Controls motion, expression, pose adherence, and background blending
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 77, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "video_frame_offset": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "expression_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "pose_adherence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "background_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "reference_image": ("IMAGE",),
                "face_video": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "background_video": ("IMAGE",),
                "character_mask": ("MASK",),
                "continue_motion": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent", "trim_image", "video_frame_offset")
    FUNCTION = "enhance"
    CATEGORY = "Wan2.2AnimateEnhancer"

    def enhance(self, positive, negative, vae, width, height, length, batch_size, 
                continue_motion_max_frames, video_frame_offset,
                motion_strength=1.0, expression_strength=1.0, 
                pose_adherence=1.0, background_blend=1.0, enable=True,
                reference_image=None, clip_vision_output=None, 
                face_video=None, pose_video=None, continue_motion=None, 
                background_video=None, character_mask=None) -> Tuple:
        
        trim_to_pose_video = False
        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        trim_latent = 0

        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3))

        image = comfy.utils.common_upscale(
            reference_image[:length].movedim(-1, 1), width, height, "area", "center"
        ).movedim(1, -1)
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.zeros(
            (1, 4, concat_latent_image.shape[-3], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), 
            device=concat_latent_image.device, dtype=concat_latent_image.dtype
        )
        trim_latent += concat_latent_image.shape[2]
        ref_motion_latent_length = 0

        if continue_motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset -= continue_motion.shape[0]
            video_frame_offset = max(0, video_frame_offset)
            continue_motion = comfy.utils.common_upscale(
                continue_motion[-length:].movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            image = torch.ones(
                (length, height, width, continue_motion.shape[-1]), 
                device=continue_motion.device, dtype=continue_motion.dtype
            ) * 0.5
            image[:continue_motion.shape[0]] = continue_motion
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        if pose_video is not None:
            if pose_video.shape[0] > video_frame_offset:
                pose_video = pose_video[video_frame_offset:]
                pose_video = comfy.utils.common_upscale(
                    pose_video[:length].movedim(-1, 1), width, height, "area", "center"
                ).movedim(1, -1)
                
                if not trim_to_pose_video and pose_video.shape[0] < length:
                    pose_video = torch.cat(
                        (pose_video,) + (pose_video[-1:],) * (length - pose_video.shape[0]), dim=0
                    )

                pose_video_latent = vae.encode(pose_video[:, :, :, :3])
                
                if enable and pose_adherence != 1.0:
                    positive = node_helpers.conditioning_set_values(
                        positive, {"pose_video_latent": pose_video_latent, "pose_strength_scale": pose_adherence}
                    )
                    negative = node_helpers.conditioning_set_values(
                        negative, {"pose_video_latent": pose_video_latent, "pose_strength_scale": pose_adherence}
                    )
                else:
                    positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": pose_video_latent})
                    negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": pose_video_latent})

                if trim_to_pose_video:
                    latent_length = pose_video_latent.shape[2]
                    length = latent_length * 4 - 3
                    image = image[:length]

        if face_video is not None:
            if face_video.shape[0] > video_frame_offset:
                face_video = face_video[video_frame_offset:]
                face_video = comfy.utils.common_upscale(
                    face_video[:length].movedim(-1, 1), 512, 512, "area", "center"
                ) * 2.0 - 1.0
                face_video = face_video.movedim(0, 1).unsqueeze(0)
                
                if enable and expression_strength != 1.0:
                    face_video_scaled = (face_video - 0.0) * expression_strength + 0.0
                    face_video_scaled = torch.clamp(face_video_scaled, -1.0, 1.0)
                    
                    positive = node_helpers.conditioning_set_values(
                        positive, {"face_video_pixels": face_video_scaled, "expression_strength_scale": expression_strength}
                    )
                    negative = node_helpers.conditioning_set_values(
                        negative, {"face_video_pixels": face_video_scaled * 0.0 - 1.0, "expression_strength_scale": 1.0}
                    )
                else:
                    positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_video})
                    negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_video * 0.0 - 1.0})

        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        if background_video is not None:
            if background_video.shape[0] > video_frame_offset:
                background_video = background_video[video_frame_offset:]
                background_video = comfy.utils.common_upscale(
                    background_video[:length].movedim(-1, 1), width, height, "area", "center"
                ).movedim(1, -1)
                
                if background_video.shape[0] > ref_images_num:
                    if enable and background_blend < 1.0:
                        neutral = torch.ones_like(background_video) * 0.5
                        background_video = background_video * background_blend + neutral * (1 - background_blend)
                    image[ref_images_num:background_video.shape[0]] = background_video[ref_images_num:]

        mask_refmotion = torch.ones(
            (1, 1, latent_length * 4, concat_latent_image.shape[-2], concat_latent_image.shape[-1]), 
            device=mask.device, dtype=mask.dtype
        )
        
        if continue_motion is not None:
            mask_refmotion[:, :, :ref_motion_latent_length * 4] = 0.0

        if character_mask is not None:
            if character_mask.shape[0] > video_frame_offset or character_mask.shape[0] == 1:
                if character_mask.shape[0] == 1:
                    character_mask = character_mask.repeat((length,) + (1,) * (character_mask.ndim - 1))
                else:
                    character_mask = character_mask[video_frame_offset:]
                    
                if character_mask.ndim == 3:
                    character_mask = character_mask.unsqueeze(1).movedim(0, 1)
                if character_mask.ndim == 4:
                    character_mask = character_mask.unsqueeze(1)
                    
                character_mask = comfy.utils.common_upscale(
                    character_mask[:, :, :length], concat_latent_image.shape[-1], 
                    concat_latent_image.shape[-2], "nearest-exact", "center"
                )
                
                if character_mask.shape[2] > ref_images_num:
                    mask_refmotion[:, :, ref_images_num:character_mask.shape[2]] = character_mask[:, :, ref_images_num:]

        concat_latent_image = torch.cat((concat_latent_image, vae.encode(image[:, :, :, :3])), dim=2)

        mask_refmotion = mask_refmotion.view(
            1, mask_refmotion.shape[2] // 4, 4, mask_refmotion.shape[3], mask_refmotion.shape[4]
        ).transpose(1, 2)
        mask = torch.cat((mask, mask_refmotion), dim=2)
        
        if enable and motion_strength != 1.0:
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask, "motion_strength_scale": motion_strength}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask, "motion_strength_scale": 1.0}
            )
        else:
            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        latent = torch.zeros(
            [batch_size, 16, latent_length + trim_latent, latent_height, latent_width], 
            device=comfy.model_management.intermediate_device()
        )
        out_latent = {"samples": latent}
        
        return (positive, negative, out_latent, trim_latent, max(0, ref_motion_latent_length * 4 - 3), video_frame_offset + length)


class WanAnimateModelEnhancer:
    """
    Model enhancer for applying motion strength control
    Place between model loader and WanAnimateToVideoEnhanced
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "enhance"
    CATEGORY = "Wan2.2AnimateEnhancer"

    def enhance(self, model, enable=True) -> Tuple:
        if not enable:
            return (model,)
        
        model_clone = model.clone()
        model_obj = model_clone.model
        
        if hasattr(model_obj, 'diffusion_model'):
            model_obj = model_obj.diffusion_model
            
        if not hasattr(model_obj, 'face_adapter'):
            print("[WanAnimateModelEnhancer] Warning: Model missing face_adapter, skipping")
            return (model,)
        
        if not hasattr(model_obj, '_original_forward_orig'):
            model_obj._original_forward_orig = model_obj.forward_orig
        
        def enhanced_forward_orig(self, x, t, context, clip_fea=None, pose_latents=None, 
                                face_pixel_values=None, freqs=None, transformer_options={}, **kwargs):
            from comfy.ldm.wan.model import sinusoidal_embedding_1d
            
            motion_scale = 1.0
            if "model_conds" in transformer_options:
                model_conds = transformer_options["model_conds"]
                if isinstance(model_conds, dict) and "motion_strength_scale" in model_conds:
                    motion_scale = model_conds["motion_strength_scale"]
            
            x = self.patch_embedding(x.float()).to(x.dtype)
            x, motion_vec = self.after_patch_embedding(x, pose_latents, face_pixel_values)
            grid_sizes = x.shape[2:]
            x = x.flatten(2).transpose(1, 2)

            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
            e = e.reshape(t.shape[0], -1, e.shape[-1])
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

            full_ref = None
            if self.ref_conv is not None:
                full_ref = kwargs.get("reference_latent", None)
                if full_ref is not None:
                    full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                    x = torch.concat((full_ref, x), dim=1)

            context = self.text_embedding(context)

            context_img_len = None
            if clip_fea is not None:
                if self.img_emb is not None:
                    context_clip = self.img_emb(clip_fea)
                    context = torch.concat([context_clip, context], dim=1)
                context_img_len = clip_fea.shape[-2]

            patches_replace = transformer_options.get("patches_replace", {})
            blocks_replace = patches_replace.get("dit", {})

            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], 
                                         freqs=args["pe"], context_img_len=context_img_len, 
                                         transformer_options=args["transformer_options"])
                        return out
                    out = blocks_replace[("double_block", i)](
                        {"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, 
                        {"original_block": block_wrap}
                    )
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, 
                             transformer_options=transformer_options)

                if i % 5 == 0 and motion_vec is not None:
                    motion_signal = self.face_adapter.fuser_blocks[i // 5](x, motion_vec)
                    scaled_motion = motion_signal * motion_scale
                    x = x + scaled_motion

            x = self.head(x, e)

            if full_ref is not None:
                x = x[:, full_ref.shape[1]:]

            x = self.unpatchify(x, grid_sizes)
            return x
        
        model_obj.forward_orig = types.MethodType(enhanced_forward_orig, model_obj)
        
        return (model_clone,)