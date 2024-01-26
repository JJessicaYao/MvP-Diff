import argparse, os, sys, glob
# from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from einops import repeat
# from main import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from imwatermark import WatermarkEncoder
import cv2
from diffusers import StableDiffusionInpaintPipeline,UNet2DConditionModel
import jsonlines
# from daam import trace
# import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf


def dense_crf(img, probs, n_labels=2):
        h = probs.shape[0]
        w = probs.shape[1]

        probs = np.expand_dims(probs, 0)
        probs = np.append(1 - probs, probs, axis=0)

        d = dcrf.DenseCRF2D(w, h, n_labels)
        U = -np.log(probs)
        U = U.reshape((n_labels, -1))
        U = np.ascontiguousarray(U)
        img = np.ascontiguousarray(img)

        U = U.astype(np.float32)
        d.setUnaryEnergy(U) 

        d.addPairwiseGaussian(sxy=2, compat=3) 
        d.addPairwiseBilateral(sxy=2, srgb=5, rgbim=img, compat=3)

        Q = d.inference(1)
        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

        return Q


def remove_area(mask):
    contours, _ = cv2.findContours(np.uint8(mask*255), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  
    cv_contours = []
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
    max_area = np.max(areas)
    idx = np.where(areas<(max_area/10))[0]

    fill = cv2.cvtColor(np.uint8(mask*255),code=cv2.COLOR_GRAY2RGB)
    for id in idx:
        # otsu1 = cv2.fillPoly(otsu1, contours[id], (0, 255, 255))
        cv2.drawContours(fill, contours, id, (0,0,0), -1)
    return fill

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def sim_mask(fault,normal,bbox):
    cos = torch.nn.CosineSimilarity(dim=2, eps=1e-7)
    seg_map = 1-cos(torch.Tensor(np.array(fault)),torch.Tensor(np.array(normal)))
    seg_map=seg_map.squeeze().cpu().numpy()
    norm_seg_map=min_max_norm(seg_map)/0.5
    norm_seg_map=norm_seg_map*np.array(bbox)
    thr,otsu=cv2.threshold(np.uint8(norm_seg_map),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.all(otsu == 0):
        return otsu
    mask = remove_area(otsu)
    mask = cv2.cvtColor(np.uint8(mask*255),code=cv2.COLOR_RGB2GRAY)
    probs = torch.sigmoid(torch.tensor(mask/255))
    mask = dense_crf(np.array(fault).astype(np.uint8), probs)
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        nargs="?",
        help="dir of dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="number of inference images",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="number of inference images",
    )
    parser.add_argument(
        "--classname",
        type=str,
        nargs="?",
        help="class name need to generate",
    )
    parser.add_argument(
        "--extract_mask",
        type=bool,
        help="whether to extract precise mask of the defective region",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="indicates extent to transform the reference image",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
    )
    opt = parser.parse_args()
    

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        opt.checkpoint_dir, 
        torch_dtype=torch.float32,
        safety_checker = None,
        requires_safety_checker = False
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline = pipeline.to(device)

    # load attention processors
    pipeline.load_lora_weights(opt.lora_dir+opt.classname)

    bks = os.listdir('./dataset/%s/ground_truth/'%opt.classname)
    num_per_bk = opt.num // bks
    h, w = 512, 512
    
    with jsonlines.open('./script/%s_test.jsonl'%opt.classname, 'r') as reader:
        for line in reader:
            os.makedirs(os.path.join(opt.output,opt.classname,'gen_img',line['mask'].split('/')[-2]), exist_ok=True)
            if opt.extract_mask:
                os.makedirs(os.path.join(opt.output,opt.classname,'prs_mask',line['mask'].split('/')[-2]), exist_ok=True)

            fls = os.listdir(os.path.join(opt.output,opt.classname,'gen_img',line['mask'].split('/')[-2]))
            if len(fls) > num_per_bk:
                continue
            
            image = line['good']
            mask = line['mask']
            prompt = line['text']
            
            image = Image.open(image)
            image = image.resize((w, h))

            mask = Image.open(mask)
            mask = mask.resize((w, h))

            
            generator = torch.Generator(device=device)
            generator = generator.manual_seed(opt.seeds)

            result = pipeline(
                prompt = prompt,
                image = image,
                mask_image = mask,
                num_inference_steps = opt.num_inference_steps,
                guidance_scale = opt.guidance_scale,
                generator = generator,
                num_images_per_prompt = 1,
                strength = opt.strength
            )

            for img in result.images:
                print('save model to ', os.path.join(opt.output,opt.classname,'gen_img',line['mask'].split('/')[-2],line['mask'].split('/')[-1]))
                img.save(os.path.join(opt.output,opt.classname,'gen_img',line['mask'].split('/')[-2],line['mask'].split('/')[-1]))
                if opt.extract_mask:
                    m = sim_mask(np.array(img),np.array(image),np.array(mask))
                    Image.fromarray(np.uint8(m*255)).save(os.path.join(opt.output,opt.classname,'prs_mask',line['mask'].split('/')[-2],line['mask'].split('/')[-1]))
              
