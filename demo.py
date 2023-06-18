import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.glip_inference import glip_inference, load_model
from src.gen_superpoint import gen_superpoint
from src.bbox2seg import bbox2seg

def Infer(input_pc_file, category, part_names, zero_shot=False, save_dir="tmp"):
    if zero_shot:
        config ="GLIP/configs/glip_Swin_L.yaml"
        weight_path = "models/glip_large_model.pth"
        print("-----Zero-shot inference of %s-----" % input_pc_file)
    else:
        config ="GLIP/configs/glip_Swin_L_pt.yaml"
        weight_path = "models/%s.pth" % category
        print("-----Few-shot inference of %s-----" % input_pc_file)
        
    print("[loading GLIP model...]")
    glip_demo = load_model(config, weight_path)

    print("[creating tmp dir...]")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)
    
    print("[normalizing input point cloud...]")
    xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
    
    print("[rendering input point cloud...]")
    img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, save_dir, device)
    
    print("[glip infrence...]")
    preds = glip_inference(glip_demo, save_dir, part_names)
    
    print('[generating superpoints...]')
    superpoint = gen_superpoint(xyz, rgb, visualize=True, save_dir=save_dir)
    
    print('[converting bbox to 3D segmentation...]')
    sem_seg, ins_seg = bbox2seg(xyz, superpoint, preds, screen_coords, pc_idx, part_names, save_dir, solve_instance_seg=True)
    
    print("[finish!]")
    
if __name__ == "__main__":
    partnete_meta = json.load(open("PartNetE_meta.json")) 
    for ins in ["Chair", "Suitcase", "Refrigerator", "Lamp", "Kettle"]:   
        Infer('examples/%s.ply' % ins, ins, partnete_meta[ins], zero_shot=True, save_dir='examples/zeroshot_%s' % ins)
        Infer('examples/%s.ply' % ins, ins, partnete_meta[ins], zero_shot=False, save_dir='examples/fewshot_%s' % ins)
        