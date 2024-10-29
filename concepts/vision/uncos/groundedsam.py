import os
import numpy as np
import torch
import torchvision
from PIL import Image
from contextlib import contextmanager, redirect_stderr, redirect_stdout

# segment anything
from segment_anything import SamPredictor

import torchvision.transforms as TS
import matplotlib.pyplot as plt

# Grounding DINO
import groundingdino.datasets.transforms as T
import groundingdino.config.GroundingDINO_SwinT_OGC as GroundingDINO_SwinT_OGC
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from .uncos_utils import MaskWrapper


class GroundedSAM(object):
    # refac from https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/automatic_label_ram_demo.py

    def __init__(self, grounding_dino_ckpt_path, box_thr, text_thr, loaded_sam):
        config_file = GroundingDINO_SwinT_OGC.__file__

        self.box_threshold = box_thr #0.3
        self.text_threshold = text_thr #0.05
        self.iou_threshold = 0.5
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # load model
        print('Loading Grounding DINO from checkpoint', grounding_dino_ckpt_path)
        with suppress_stdout_stderr():
            self.model = self.load_model(config_file, grounding_dino_ckpt_path)
        print('Loaded Grounding DINO')
        self.model = self.model.to(self.device)
        self.sam_predictor = SamPredictor(loaded_sam)

        # initialize Recognize Anything Model
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), normalize
        ])

    def load_image(self, image_rgb_255):
        # load image
        image_pil = Image.fromarray(np.uint8(image_rgb_255))

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_normalized, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image_normalized

    def process_image(self, image_rgb_255, text_prompt):
        # load image
        image_pil, image_normalized = self.load_image(image_rgb_255)
        image_rgb_255 = np.array(image_pil)
        tags=text_prompt
        # run grounding dino model
        boxes_filt, scores, pred_phrases = self.get_grounding_output(image_normalized, tags)

        self.sam_predictor.set_image(image_rgb_255)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        if len(boxes_filt)==0:
            return []
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image_rgb_255.shape[:2]).to(self.device)

        masks, iou_predictions, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        mask_np = [mask.astype(bool) for mask in masks[:,0].detach().cpu().numpy()]
        score_np = iou_predictions[:,0].detach().cpu().numpy()
        return [MaskWrapper({'segmentation': mask, 'predicted_iou': score, 'bbox': box.numpy()})
                for (mask, score, box) in zip(mask_np, score_np, boxes_filt)]

    def load_model(self, model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        # print(load_res)
        _ = model.eval()
        return model

    def get_grounding_output(self, image, caption):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(self.device)
        with torch.no_grad(), suppress_stdout_stderr():
            outputs = self.model(image[np.newaxis], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        # logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        # logits_filt.shape[0]

        # get phrase
        tokenlizer = self.model.tokenizer
        with suppress_stdout_stderr():
            tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
