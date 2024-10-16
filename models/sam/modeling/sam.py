# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from einops import rearrange
import torch.nn.functional as F
from conf import settings
from utils import *
from monai.metrics import compute_hausdorff_distance, DiceMetric
from monai.losses import DiceCELoss
from pathlib import Path

import pandas as pd

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        args,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an images and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            images into images embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the images embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input images.
          pixel_std (list(float)): Std values for normalizing pixels in the input images.
        """
        super().__init__()
        self.args = args
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.pt = None
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        imgs# batched_input: List[Dict[str, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'images': The images as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the images before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this images, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the images.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        multimask_output = False
        # input_images = torch.stack([self.preprocess(x["images"]) for x in batched_input], dim=0)
        # input_images = imgs
        # image_embeddings = self.image_encoder(input_images)
        outputs = []

        args = cfg.parse_args()
        if args.mod == 'sam_adpt':
                for n, value in self.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = True
                    else:
                        value.requires_grad = True
        else:
            for n, value in self.image_encoder.named_parameters():
                value.requires_grad = True
        imgs = imgs.requires_grad_(True)
        imge = self.image_encoder(imgs).requires_grad_(True)
        # print(imge)
        pt = self.pt

        if args.net == 'sam' or args.net == 'mobile_sam':
            se, de = self.prompt_encoder(
                points=pt,
                boxes=None,
                masks=None,
            )
        elif args.net == "efficient_sam":
            coords_torch ,labels_torch = transform_prompt(coords_torch ,labels_torch ,h ,w)
            se = self.prompt_encoder(
                coords=coords_torch,
                labels=labels_torch,
            )
        if args.net == 'sam' or args.net == 'mobile_sam':
            pred, _ = self.mask_decoder(
                image_embeddings=imge,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )
            print(pred)
        elif args.net == "efficient_sam":
            se = se.view(
                se.shape[0],
                1,
                se.shape[1],
                se.shape[2],
            )
            pred, _ = self.mask_decoder(
                image_embeddings=imge,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                multimask_output=False,
            )
        # print(pred)
        # Resize to the ordered output size
        pred = F.interpolate(pred ,size=(args.out_size ,args.out_size))
        # print(pred)
        return pred

        # for image_record, curr_embedding in zip(batched_input, image_embeddings):
        #
        #
        #     if "point_coords" in image_record:
        #         points = (image_record["point_coords"], image_record["point_labels"])
        #     else:
        #         points = None
        #
        #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #         points=points,
        #         boxes=image_record.get("boxes", None),
        #         masks=image_record.get("mask_inputs", None),
        #     )
        #     low_res_masks, iou_predictions = self.mask_decoder(
        #         image_embeddings=curr_embedding.unsqueeze(0),
        #         image_pe=self.prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=multimask_output,
        #     )
        #     masks = self.postprocess_masks(
        #         low_res_masks,
        #         input_size=image_record["images"].shape[-2:],
        #         original_size=image_record["original_size"],
        #     )
        #     masks = masks > self.mask_threshold
        #     outputs.append(
        #         {
        #             "masks": masks,
        #             "iou_predictions": iou_predictions,
        #             "low_res_logits": low_res_masks,
        #         }
        #     )
        # return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original images size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the images input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the images
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    # def forward(
    #         self,
    #         imgs):
    #     imge = self.image_encoder(imgs)
    #     pt = None
    #     se, de = self.prompt_encoder(
    #         points=pt,
    #         boxes=None,
    #         masks=None,
    #     )
    #
    #
    #     pred, _ = self.mask_decoder(
    #         image_embeddings=imge,
    #         image_pe=self.prompt_encoder.get_dense_pe(),
    #         sparse_prompt_embeddings=se,
    #         dense_prompt_embeddings=de,
    #         multimask_output=False,
    #     )
    #
    #     # print(pred.shape)
    #     # Resize to the ordered output size
    #     pred = F.interpolate(pred, size=(1024, 1024))
    #     return pred
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
