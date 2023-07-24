# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import json
from typing import Any
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from typing import List

import cv2
import matplotlib.pyplot as plt
from cog import BasePredictor, Input, Path, BaseModel

from subprocess import call

HOME = os.getcwd()
os.chdir("GroundingDINO")
call("pip install -q .", shell=True)
os.chdir(HOME)

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# segment anything
from segment_anything import build_sam, SamPredictor



class ModelOutput(BaseModel):
    image_mask_path: List[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.image_size = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


        self.model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "pretrained/groundingdino_swint_ogc.pth",
            device=self.device,
        )

        self.sam = SamPredictor(
            build_sam(checkpoint="pretrained/sam_vit_h_4b8939.pth").to(self.device)
        )

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        object_prompt: str = Input(description="Object prompt"),
        box_threshold: float = Input(default=0.25, description="Box threshold"),
        text_threshold: float = Input(default=0.2, description="Text threshold"),
        dilation: int = Input(default=0, description="Dilation"),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        print("Running prediction")
        image_source, image = load_image(str(input_image))
        print("Loaded image")
        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.model, image, object_prompt, box_threshold, text_threshold
        )
        print("Got grounding output")

        self.sam.set_image(image)
        size = image_source.size
        H, W = size[1], size[0]
        print("H, W", H, W)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_filt) * torch.Tensor([W, H, W, H])
        print("boxes_xyxy", boxes_xyxy)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(self.device)


        masks, _, _ = self.sam.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
        print("Got masks")
        image_masks = [i.cpu().numpy() for i in masks]
        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            image_masks = [cv2.dilate(i, kernel, iterations=1) for i in image_masks]
        image_mask_pils = [Image.fromarray(i) for i in image_masks]
        image_mask_pil_paths = []
        # save to tmp file
        for i, image_mask_pil in enumerate(image_mask_pils):
            image_mask_pil.save(f"/tmp/image_mask{i}.png")
            image_mask_pil_paths.append(Path(f"/tmp/image_mask{i}.png"))
        print("Saved image mask")
        return ModelOutput(image_mask_path=image_mask_pil_paths)


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, device="cpu"
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenizer
        )
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=1.5)
    )
    ax.text(x0, y0, label)