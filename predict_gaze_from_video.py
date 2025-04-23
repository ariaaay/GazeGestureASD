# %%
import os
import pickle
import argparse
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

import torch
from retinaface import RetinaFace
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# import tensorflow as tf
# print("GPUs Available:", tf.config.list_physical_devices('GPU'))
# print(tf.sysconfig.get_build_info()["cudnn_version"])


def extract_video_paths(data_folder, session_key):
    # iterate through folder to extract video and annotation files
    video_files = []
    for subj in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, subj)):
            for sess in os.listdir(os.path.join(data_folder, subj)):
                if sess == session_key:
                    for file in os.listdir(os.path.join(data_folder, subj, sess)):
                        if file.endswith(".wmv"):
                            video_files.append(os.path.join(data_folder, subj, sess, file))
    return video_files
    
def convert_frame_to_images(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    return image_pil

# visualize predicted gaze heatmap for each person and gaze in/out of frame score
def detect_eye_contact(heatmap, bboxes, width, height, normalize_heatmap):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    heatmap = np.asarray(heatmap).copy()
    if normalize_heatmap:
        heatmap = heatmap / np.sum(heatmap)
    scores = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        scores.append(np.sum(heatmap[int(ymin*height):int(ymax*height), int(xmin*width):int(xmax*width)]) / np.sum(heatmap))
    return scores

def rearrange_bbox_l2r(bboxes):
    # rearrange bboxes so it is from left to right on image
    bboxes = np.array(bboxes)
    order = np.argsort(bboxes[:, 0])
    bboxes = bboxes[order]
    return list(bboxes), order
    
    
def visualize_heatmap(pil_image, heatmap, bbox=None, inout_score=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(90)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="lime", width=int(min(width, height) * 0.01))

        if inout_score is not None:
          text = "%.2f" % inout_score
          text_width = draw.textlength(text)
          text_height = int(height * 0.01)
          text_x = xmin * width
          text_y = ymax * height + text_height
          draw.text((text_x, text_y), text, fill="lime", font=ImageFont.load_default(size=int(min(width, height) * 0.05)))
    return overlay_image
# combined visualization with maximal gaze points for each person

def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
    bboxes, l2r_order = rearrange_bbox_l2r(bboxes)
    heatmaps = heatmaps[l2r_order]
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=int(min(width, height) * 0.01))
        if inout_scores is not None:
            inout_score = inout_scores[i]
            other_bboxes = bboxes.copy()
            other_bboxes.pop(i)
            eye_contact_scores = detect_eye_contact(heatmaps[i], other_bboxes, width, height, normalize_heatmap=True)
            if i == 1: # TODO: eye contact is true as long as it is between child and others. Simplify this when no plotting needed
                children_eye_contact_score = max(eye_contact_scores)
            text = "%.2f" % inout_score + "\n" + "|".join(["%.2f" % score for score in eye_contact_scores])
            # text = "%.2f" % inout_score
            # text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill=color, font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

        if inout_scores is not None and inout_score > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap.detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            draw.ellipse([(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], fill=color, width=int(0.005*min(width, height)))
            draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=int(0.005*min(width, height)))
    
    if len(bboxes) < 3: # if parent, experimenter or children not in frame. if change this you need to reevalute how you pick the children bbox
        children_eye_contact_score = -1

    return overlay_image, children_eye_contact_score

def visualize_gaze_in_image(model, transform, images):
    gaze_scores = []
    overlay_images = []
    width, height = images[0].size
    print("Number of images:", len(images))
    print("Image size:", width, height)
    
    norm_bboxes = []
    img_tensors = []
    fail_indexes = []
    for i, image in enumerate(images):
        resp = RetinaFace.detect_faces(np.array(image), threshold=0.5)
        bboxes = [resp[key]['facial_area'] for key in resp.keys()]
        img_tensors.append(transform(image).unsqueeze(0).to(device))
        norm_bboxes.append([np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes])
        
    input = {
        "images": torch.cat(img_tensors, 0), # [num_images, 3, 448, 448]
        "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
    }
    print(norm_bboxes)
    print(len(norm_bboxes))
    
    with torch.no_grad():
        output = model(input)
        
        
    for i, image in enumerate(images):
        if i in fail_indexes:
            overlay_images.append(image)
            gaze_scores.append(-1)
        else:
            vis, score = visualize_all(image, output['heatmap'][i], norm_bboxes[i], output['inout'][i] if output['inout'] is not None else None, inout_thresh=0.5)
            overlay_images.append(vis)
            gaze_scores.append(score)
    
        # if plotting:
        #     for b in range(len(bboxes)):
        #         plt.figure()
        #         plt.imshow(visualize_heatmap(image, output['heatmap'][i][b], norm_bboxes[i][b], inout_score=output['inout'][[i]][b] if output['inout'] is not None else None))
        #         plt.axis('off')
        #         plt.show()
    
    return np.array(gaze_scores), overlay_images
# %%
def extract_frame_at_time(video_path, time_secs):
    imgs = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    # Get the frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"Video FPS: {fps}, Total frames: {total_frames}, Duration: {duration_sec:.2f}s")

    for time_sec in time_secs:
        if time_sec > duration_sec:
            raise ValueError("Requested time exceeds video duration")
        # Calculate the frame index to extract
        frame_number = int(fps * time_sec)
        # Set the video position to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        img = convert_frame_to_images(frame)
        imgs.append(img)
        if not ret:
            raise RuntimeError("Failed to read the frame at the specified time")
    cap.release()
    return imgs

def code_gaze_label(anns):
    labels = []
    for item in anns:
        if item == "eye contact":
            labels.append(1)
        elif item == "no eye contact":
            labels.append(0)
        else:
            break
    return np.array(labels)

# video_file = "/Datasets/.Autism_Videos/data/bids/sub-9943/ses-18mo/sub-9943_ses-18mo_measure-CSBS_video.wmv"
# xlsx_file_path = "/Datasets/.Autism_Videos/data/bids/sub-9943/ses-18mo/sub-9943_ses-18mo_measure-CSBS_coding.xlsx"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze prediction from video")
    parser.add_argument("--plotting", action="store_true", help="Plot the gaze prediction results")
    parser.add_argument("--data_folder", type=str, default="/Datasets/.Autism_Videos/data/bids/", help="Path to the data folder")
    parser.add_argument("--session_key", type=str, default="ses-18mo", help="Session key to filter videos")
    parser.add_argument("--video_path", type=str, help="Path to specific video file")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to process")
    args = parser.parse_args()
    data_folder = args.data_folder

    try:
        video_files = pickle.load(open("output/%s_video_files.pkl" % args.session_key, "rb"))
    except FileNotFoundError:
        video_files = extract_video_paths(data_folder, args.session_key)
        pickle.dump(video_files, open("output/%s_video_files.pkl" % args.session_key, "wb"))                          
                            
    if args.video_path is not None:
        video_list = [args.video_path]
    else:
        video_list = video_files[:args.num_samples] # just a sample for testing
        
    # load Gaze-LLE model
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    model.eval()
    model.to(device)

    accs = []
    for video_file in video_list:
        print(video_file)
        # extract video name from path
        annotation_file = video_file.replace("video.wmv", "coding.xlsx")
        try:
            anns = pd.read_excel(annotation_file)
        except FileNotFoundError:
            continue
        gaze_label = code_gaze_label(anns["EYE CONTACT"])
        label_len = len(gaze_label) # get rid of the NaN values
        begin_times = anns["Begin Time - ss.msec"][:label_len]
        end_times = anns["End Time - ss.msec"][:label_len] 
        sample_times = (begin_times + end_times) / 2
        imgs = extract_frame_at_time(video_file, sample_times)
        if label_len > 0:
            gaze_scores, overlay_images = visualize_gaze_in_image(model, transform, imgs)
            print(gaze_scores)
            print(gaze_label)
            accs.append(np.mean((gaze_scores>0.2).astype(int) == gaze_label))
            # TODO: calculate false positive and false negative
        
            if args.plotting:
                for i in range(label_len):
                    plt.figure(figsize=(10,10))
                    plt.imshow(overlay_images[i])
                    plt.axis('off')
                    plt.savefig(os.path.join("figures", os.path.basename(video_file).replace(".wmv", "_%d.png" % i)), bbox_inches='tight', pad_inches=0)
                    
        
        
    print(accs)
    print("Mean accuracy:", np.mean(accs))
    np.save("output/gaze_accs.npy", accs)
        
                
            