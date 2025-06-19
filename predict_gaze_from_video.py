# %%
import os
import pickle
import argparse
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data._utils.collate import default_collate

from retinaface import RetinaFace
from deep_sort_pytorch.deep_sort import DeepSort
import tensorflow as tf
import gc
from tensorflow.keras import backend as K
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# if torch.cuda.is_available():
#     device = "cuda:1"
# # print("GPUs Available:", tf.config.list_physical_devices('GPU'))
# # print(tf.sysconfig.get_build_info()["cudnn_version"])
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)



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
    bboxes = bboxes.cpu().numpy() if isinstance(bboxes, torch.Tensor) else np.array(bboxes)
    order = np.argsort(bboxes[:, 0])
    bboxes = bboxes[order]
    return list(bboxes), order

def cluster_bbox_for_child_pos(bboxes):
    from sklearn.cluster import KMeans
    # cluster bboxes to find three clusters
    all_centers = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for bbox in bboxes for b in bbox]
    # import kmeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(all_centers)
    centeroids = kmeans.cluster_centers_
    order = np.argsort(centeroids[:, 0])
    return centeroids[order][1]
    
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

def calculate_bbox_distance_to_child_pos(point, bboxes):
    # calculate distance between point and bboxes
    distances = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        distance = np.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)
        distances.append(distance)
    return distances

def visualize_all_with_tracking(pil_image, heatmaps, bboxes, inout_scores, ds_output, child_ids, inout_thresh=0.5):
    # computing gaze for 1 image
    ds_output = np.array(ds_output)
    bboxes, l2r_order = rearrange_bbox_l2r(bboxes)
    heatmaps = heatmaps[l2r_order]
    
    child_ind = -1
    if len(bboxes) == 3:
        child_ind = 1
    else:
        if len(ds_output) > 0: # TODO: should i worried about cases where len > 3?
            ds_output = ds_output[l2r_order]
            track_ids = ds_output[:, -1]
            for i, tid in enumerate(track_ids):
                if tid in child_ids:
                    child_ind = i
                    # print(len(track_ids))
                    # print("using tracking IDs, child pos %d" % i)
        
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size
    children_eye_contact_score = -1
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=int(min(width, height) * 0.01))
        if i == child_ind:
            if inout_scores is not None:
                inout_score = inout_scores[i]
                other_bboxes = bboxes.copy()
                other_bboxes.pop(i)
                eye_contact_scores = detect_eye_contact(heatmaps[i], other_bboxes, width, height, normalize_heatmap=True)
                if len(eye_contact_scores) > 0:
                    children_eye_contact_score = max(eye_contact_scores)
            text = "%.2f" % children_eye_contact_score
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
    # print(children_eye_contact_score)
    return overlay_image, children_eye_contact_score

def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5, child_pos=None):
    bboxes, l2r_order = rearrange_bbox_l2r(bboxes)
    heatmaps = heatmaps[l2r_order]
    
    if len(bboxes) == 3:
        child_ind = 1
    else:
        if child_pos is not None:
            distances = calculate_bbox_distance_to_child_pos(child_pos, bboxes)
            child_ind = np.argmin(distances)
        else:
            child_ind = -1
        
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size
    children_eye_contact_score = -1
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=int(min(width, height) * 0.01))
        if i == child_ind:
            if inout_scores is not None:
                inout_score = inout_scores[i]
                other_bboxes = bboxes.copy()
                other_bboxes.pop(i)
                eye_contact_scores = detect_eye_contact(heatmaps[i], other_bboxes, width, height, normalize_heatmap=True)
                if len(eye_contact_scores) > 0:
                    children_eye_contact_score = max(eye_contact_scores)
            text = "%.2f" % inout_score + "|" + "%.2f" % children_eye_contact_score
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

    return overlay_image, children_eye_contact_score

def compute_eye_contact_only(pil_image, heatmaps, bboxes, inout_scores, child_pos):
    distances = calculate_bbox_distance_to_child_pos(child_pos, bboxes)
    child_ind = np.argmin(distances)
    width, height = pil_image.size
    children_eye_contact_score = -1
    
    if inout_scores is not None:
        other_bboxes = bboxes.copy()
        other_bboxes.pop(child_ind)
        eye_contact_scores = detect_eye_contact(heatmaps[child_ind], other_bboxes, width, height, normalize_heatmap=True)
        if len(eye_contact_scores) > 0:
            children_eye_contact_score = max(eye_contact_scores)
        
    return children_eye_contact_score

def estimate_face_bboxes(images, transform, face_threshold=0.5):
    width, height = images[0].size
    # print("Image size:", width, height)
    norm_bboxes = []
    img_tensors = []
    fail_indexes = []
    for i, image in enumerate(images):
        resp = RetinaFace.detect_faces(np.array(image), threshold=face_threshold)
        bboxes = [resp[key]['facial_area'] for key in resp.keys()]
        if bboxes is None or len(bboxes) == 0:
            # print("No face detected in image %d" % i)
            fail_indexes.append(i)
        else:
            img_tensors.append(transform(image).unsqueeze(0).to(device))
            norm_bboxes.append([np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes])
            
    return img_tensors, norm_bboxes, fail_indexes

def estimate_gaze_in_image(model, transform, images, batch_size=64, face_threshold=0.5):
    print("Number of images:", len(images))
    # to avoid OOM, we split the input into microbatches
    total_size = len(images)
    output_heatmap, output_inout, all_norm_bboxes = [], [], []
    
    for i in tqdm(range(0, total_size, batch_size)):
        images_batch = images[i:i+batch_size]
        img_tensors, norm_bboxes, fail_indexes = estimate_face_bboxes(images_batch, transform, face_threshold=face_threshold)
        input = {
            "images": torch.cat(img_tensors, 0), # [num_images, 3, 448, 448]
            "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
        }
        
        # After TF is done:
        K.clear_session()
        gc.collect()
        
        with torch.no_grad():
            batch_output = model(input)
    
        output_heatmap += batch_output["heatmap"]
        output_inout += batch_output["inout"]
        all_norm_bboxes += norm_bboxes

    output = {
        "heatmap": output_heatmap,
        "inout": output_inout
    }
    return images, all_norm_bboxes, output, fail_indexes

def estimate_gaze_in_image_with_tracking(model, transform, images, ds_outputs, rf_bboxes, batch_size=64):
    print("Number of images:", len(images))
    # to avoid OOM, we split the input into microbatches
    width, height = images[0].size
    total_size = len(images)
    output_heatmap, output_inout, all_norm_bboxes, fail_indexes = [], [], [], []
    
    # ds_outputs[0] = [convert_rf_bbox_to_ds_bbox(bbox) for bbox in rf_bboxes[0]] # deep sort doesnt output bbox for the first two frames
    # ds_outputs[1] = [convert_rf_bbox_to_ds_bbox(bbox) for bbox in rf_bboxes[1]] # deep sort doesnt output bbox for the first two frames
    # # print(ds_outputs[:1])
    # # print(ds_outputs[2])

    for i in tqdm(range(0, total_size, batch_size)): # every batch
        images_batch = images[i:i+batch_size]
        ds_outputs_batch = ds_outputs[i:i+batch_size]
        norm_bboxes, img_tensors = [], []
        for j, output in enumerate(ds_outputs_batch): # every image in the batch
            if len(output) == 0:
                print("No face detected in image %d" % (j+i))
                fail_indexes.append(j+i)
            else:
                img_tensors.append(transform(images_batch[j]).unsqueeze(0).to(device))
                norm_bboxes.append([convert_ds_bbox_to_rf_bbox(out[:4]) / np.array([width, height, width, height]) for out in output])
        
        input = {
            "images": torch.cat(img_tensors, 0), # [num_images, 3, 448, 448]
            "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
        }
        
        # # After TF is done:
        # K.clear_session()
        # gc.collect()
        
        with torch.no_grad():
            batch_output = model(input)
    
        output_heatmap += batch_output["heatmap"]
        output_inout += batch_output["inout"]
        all_norm_bboxes += norm_bboxes

    gazelle_output = {
        "heatmap": output_heatmap,
        "inout": output_inout
    }
    return images, all_norm_bboxes, gazelle_output, fail_indexes

def calculate_gaze_score_in_image(images, norm_bboxes, output, fail_indexes, child_pos=None):
    gaze_scores = []
    overlay_images = []
    
    k = 0
    for i, image in enumerate(tqdm(images)):
        if i in fail_indexes:
            overlay_images.append(image)
            gaze_scores.append(-1)
            k+=1
        else:
            vis, score = visualize_all(image, output['heatmap'][i-k], norm_bboxes[i-k], output['inout'][i-k] if output['inout'][i-k] is not None else None, inout_thresh=0.5, child_pos=child_pos if child_pos else None)
            # vis = visualize_heatmap(image, output['heatmap'][i-k], norm_bboxes[i-k], output['inout'][i-k] if output['inout'][i-k] is not None else None)
            overlay_images.append(vis)
            gaze_scores.append(score)
    
    return np.array(gaze_scores), overlay_images

def calculate_gaze_score_in_image_with_tracking(images, norm_bboxes, gazelle_output, fail_indexes, ds_outputs, child_ids):
    gaze_scores = []
    overlay_images = []
    
    k = 0
    for i, image in enumerate(tqdm(images)):
        if i in fail_indexes:
            overlay_images.append(image)
            gaze_scores.append(-1)
            k+=1
        else:
            vis, score = visualize_all_with_tracking(image, gazelle_output['heatmap'][i-k], norm_bboxes[i-k], gazelle_output['inout'][i-k] if gazelle_output['inout'][i-k] is not None else None, ds_outputs[i], child_ids, inout_thresh=0.5)
            overlay_images.append(vis)
            gaze_scores.append(score)
    
    return np.array(gaze_scores), overlay_images

def extract_all_frames(video_file):
    vid = cv2.VideoCapture(video_file)
    images = []
    while True:
        vid.grab()
        retval, image = vid.retrieve()
        if not retval:
            break
        image_pil = convert_frame_to_images(image)
        images.append(image_pil)
    vid.release()
    return images

def extract_frame_at_times(video_path, time_secs):
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

def extract_frame_at_time_ranges(video_path, time_ranges):
    imgs = []
    trial_tracker = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    # Get the frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"Video FPS: {fps}, Total frames: {total_frames}, Duration: {duration_sec:.2f}s")

    for i, time_range in enumerate(time_ranges):
        t_start, t_end = time_range
        if t_end > duration_sec:
            print("Requested time exceeds video duration")
            break
        # Calculate the frame index to extract
        start_frame = int(fps * t_start)
        end_frame = int(fps * t_end)
        # Set the video position to the frame
        for frame_number in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            ret, frame = cap.read()
            img = convert_frame_to_images(frame)
            imgs.append(img)
            trial_tracker.append(i)
        if not ret:
            raise RuntimeError("Failed to read the frame at the specified time")
    cap.release()
    return imgs, np.array(trial_tracker)

def extract_detect_and_track_faces(deepsort_init, video_path, time_ranges):
    all_ds_outputs, all_rf_bboxes, all_images, trial_tracker = [], [], [], []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    # Get the frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"Video FPS: {fps}, Total frames: {total_frames}, Duration: {duration_sec:.2f}s")
    for i, time_range in enumerate(time_ranges):
        deepsort = deepsort_init.copy()
        t_start, t_end = time_range
        if t_end > duration_sec:
            print("Requested time exceeds video duration")
            break
        # Calculate the frame index to extract
        start_frame = int(fps * t_start)
        end_frame = int(fps * t_end)
        # Set the video position to the frame
        fnum = 0
        for frame_number in range(start_frame-3, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
                        
            # Detect faces with RetinaFace
            results = RetinaFace.detect_faces(frame)
            bboxes = []
            confidences = []
            
            for key in results:
                face = results[key]
                x1, y1, x2, y2 = face['facial_area']
                bboxes.append([x1, y1, x2 - x1, y2 - y1])  # xywh format
                confidences.append(face['score'])

            # Convert to numpy array
            bboxes = np.array(bboxes)
            confidences = np.array(confidences)
            
            # Update Deep SORT tracker
            outputs, _ = deepsort.update(bboxes, confidences, np.zeros((len(bboxes))), frame)
            if fnum > 2: # skip first two frames which are used used for deepsort
                all_ds_outputs.append(outputs)
                all_rf_bboxes.append(bboxes)
                all_images.append(convert_frame_to_images(frame))
                trial_tracker.append(i)
            
            fnum += 1

    cap.release()
    return all_images, all_rf_bboxes, all_ds_outputs, trial_tracker

def extract_child_ids(all_ds_outputs):
    # find frame with child and get the track id of middle
    child_id = []
    for i, outputs in enumerate(all_ds_outputs):
        if len(outputs) == 3: # 3 faces
            outputs = np.array(outputs)
            order = np.argsort(outputs[:, 0]) # sort bbox by xmin
            child_id.append(outputs[order[1], -1])
    
    child_id = set(child_id)
    return list(child_id)

def convert_ds_bbox_to_rf_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    new_box = [xmin+int(w/2), ymin+int(h/2), xmax+int(w/2), ymax+int(h/2)]
    return new_box

def convert_rf_bbox_to_ds_bbox(bbox):
    # rfbox alrady in xywh format
    cxmin, cymin, w, h = bbox
    cxmax = cxmin + w
    cymax = cymin + h
    xmin = cxmin - int(w / 2)
    ymin = cymin - int(h / 2)
    xmax = cxmax - int(w / 2)
    ymax = cymax - int(h / 2)
    return [xmin, ymin, xmax, ymax]

def draw_both_bbox(image, rf_bboxes, ds_outputs, plotting=False):
    draw = ImageDraw.Draw(image)
    
    for bbox in rf_bboxes:
        xmin, ymin, width, height = bbox
        draw.rectangle([xmin, ymin, xmin+width, ymin+height], outline="lime", width=5)
        
    for i, out in enumerate(ds_outputs):
        track_id = out[-1]
        xmin, ymin, xmax, ymax = out[:4]
        new_box = convert_ds_bbox_to_rf_bbox(out[:4])
        draw.rectangle(new_box, outline="red", width=2)
        draw.text((xmin, ymin), f"ID: {track_id}", fill="red")
    
    if plotting:
        plt.figure()
        plt.imshow(image)
    return image
    # plt.show()
    
def load_gaze_model():
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    model.eval()
    model.to(device)
    return model, transform

def code_gaze_label(anns):
    labels = []
    for item in anns:
        if item == "eye contact":
            labels.append(1)
        elif item == "no eye contact":
            labels.append(0)
        else:
            continue
    return np.array(labels)

def save_model_output(output_dir, video_file, images, norm_bboxes, output, fail_indexes):
    name = video_file.split("/")[-1]
    output_file = os.path.join(output_dir, name.replace(".wmv", "_output.pkl"))
    with open(output_file, "wb") as f:
        pickle.dump({
            "images": images,
            "norm_bboxes": norm_bboxes,
            "output": output,
            "fail_indexes": fail_indexes
        }, f)
    print("Saved model output to", output_file)
    
def save_model_output_with_tracking(output_dir, video_file, all_images, norm_bboxes, gazelle_output, all_ds_outputs, fail_indexes, trial_tracker):
    save_name = video_file.split("/")[-1]
    output_file = os.path.join(output_dir, save_name.replace(".wmv", "_output_with_tracking.pkl"))
    with open(output_file, "wb") as f:
        pickle.dump({
            "images": all_images,
            "norm_bboxes": norm_bboxes,
            "gazelle_output": gazelle_output,
            "fail_indexes": fail_indexes,
            "all_ds_outputs": all_ds_outputs,
            "trial_tracker": trial_tracker
        }, f)
    print("Saved model output to", output_file)
    
def load_model_output(output_dir, video_file):
    name = video_file.split("/")[-1]
    output_file = os.path.join(output_dir, name.replace(".wmv", "_output.pkl"))
    with open(output_file, "rb") as f:
        data = pickle.load(f)
    images = data["images"]
    norm_bboxes = data["norm_bboxes"]
    output = data["output"]
    fail_indexes = data["fail_indexes"]
    print("Loaded model output from", output_file)
    return images, norm_bboxes, output, fail_indexes

def load_model_output_with_tracking(output_dir, video_file):
    name = video_file.split("/")[-1]
    output_file = os.path.join(output_dir, name.replace(".wmv", "_output_with_tracking.pkl"))
    with open(output_file, "rb") as f:
        data = pickle.load(f)
    all_images = data["images"]
    norm_bboxes = data["norm_bboxes"]
    gazelle_output = data["gazelle_output"]
    all_ds_outputs = data["all_ds_outputs"]
    fail_indexes = data["fail_indexes"]
    trial_tracker = data["trial_tracker"]
    print("Loaded model output from", output_file)
    return all_images, norm_bboxes, gazelle_output, all_ds_outputs, fail_indexes, trial_tracker

def check_output_exists(output_dir, video_file, tracking=False):
    name = video_file.split("/")[-1]
    if tracking:
        output_path = os.path.join(output_dir, name.replace(".wmv", "_output_with_tracking.pkl"))
    else: 
        output_path = os.path.join(output_dir, name.replace(".wmv", "_output.pkl"))
    return os.path.exists(output_path)

def load_annotation_file(video_file):
    # extract video name from path
    annotation_file = video_file.replace("video.wmv", "coding.xlsx")
    try:
        anns = pd.read_excel(annotation_file)
    except FileNotFoundError:
        try:
            names = annotation_file.split("/")
            new_name = names[-1][:4] + names[-1][5:]
            new_path = "/".join(names[:-1]) + "/" + new_name
            print(f"trying to load {new_path}")
            anns = pd.read_excel(new_path)
        except FileNotFoundError:
            print("Gaze label not found for %s" % video_file)
            return None, None, None
    gaze_label = code_gaze_label(anns["EYE CONTACT"])
    label_len = len(gaze_label) # get rid of the NaN values
    if label_len == 0:
        print("Gaze label length is 0 for %s" % video_file)
        return None, None, None
    else:
        begin_times = anns["Begin Time - ss.msec"][:label_len]
        end_times = anns["End Time - ss.msec"][:label_len]
        return gaze_label, begin_times, end_times

def predict_gaze_from_video(video_list, model, transform, output_dir, batch_size=64, plotting=False):
    results = []
    for video_file in video_list:
        if plotting:
            images_to_plot = []
            
        gaze_label, begin_times, end_times = load_annotation_file(video_file)
        if gaze_label is not None:
            try:
                images, norm_bboxes, output, fail_indexes = load_model_output(args.gazelle_output_dir, video_file)
                _, trial_tracker = extract_frame_at_time_ranges(video_file, list(zip(begin_times, end_times)))
            except FileNotFoundError:
                imgs, trial_tracker = extract_frame_at_time_ranges(video_file, list(zip(begin_times, end_times)))
                images, norm_bboxes, output, fail_indexes = estimate_gaze_in_image(model, transform, imgs, batch_size=batch_size, face_threshold=0.5)
                save_model_output(output_dir, video_file, images, norm_bboxes, output, fail_indexes)
            
            gaze_scores, overlay_images = calculate_gaze_score_in_image(images, norm_bboxes, output, fail_indexes)
            # print(gaze_scores)
            # print(gaze_label)
            # print(trial_tracker)
            assert len(trial_tracker) == len(gaze_scores)
            seg_score = np.zeros(len(gaze_label))
            for i in range(len(gaze_label)):
                trial_ind = trial_tracker == i
                seg_score[i], j = np.max(gaze_scores[trial_ind]), np.argmax(gaze_scores[trial_ind])
                if plotting:
                    images_to_plot.append(np.array(overlay_images)[trial_ind][j])
            gaze_pred = seg_score > 0.2
            acc = np.mean(gaze_pred == gaze_label)
            specificity = np.sum((gaze_pred == 0) & (gaze_label == 0)) / np.sum(gaze_label == 0)
            sensitivity = np.sum((gaze_pred == 1) & (gaze_label == 1)) / np.sum(gaze_label == 1)
            results.append((os.path.basename(video_file), acc, specificity, sensitivity))
            print("Accuracy:", acc)
            if plotting:
                for i in range(len(gaze_label)):
                    plt.figure(figsize=(10,10))
                    plt.imshow(images_to_plot[i])
                    plt.axis('off')
                    plt.savefig(os.path.join("figures/gaze_without_tracking", os.path.basename(video_file).replace(".wmv", "_%d.png" % i)), bbox_inches='tight', pad_inches=0)
        np.save("output/gaze_accs.npy", results) # save as it goes
    return results

def predict_gaze_from_video_with_tracking(video_list, model, transform, output_dir, batch_size=64, plotting=False):
    results = []
    for video_file in video_list:
        try:
            if plotting:
                images_to_plot = []
                
            gaze_label, begin_times, end_times = load_annotation_file(video_file)
            if gaze_label is not None:
                try:
                    all_images, norm_bboxes, gazelle_output, all_ds_outputs, fail_indexes, trial_tracker = load_model_output_with_tracking(args.gazelle_output_dir, video_file)
                    # print(len(all_ds_outputs), len(all_images), len(norm_bboxes), len(gazelle_output['heatmap']), len(gazelle_output['inout']))
                except FileNotFoundError:
                    all_images, all_rf_bboxes, all_ds_outputs, trial_tracker = extract_detect_and_track_faces(video_file, list(zip(begin_times, end_times)))
                    all_images, norm_bboxes, gazelle_output, fail_indexes = estimate_gaze_in_image_with_tracking(model, transform, all_images, all_ds_outputs, all_rf_bboxes, batch_size=batch_size)
                    save_model_output_with_tracking(output_dir, video_file, all_images, norm_bboxes, gazelle_output, all_ds_outputs, fail_indexes, trial_tracker)
                
                child_ids = extract_child_ids(all_ds_outputs)
                gaze_scores, overlay_images = calculate_gaze_score_in_image_with_tracking(all_images, norm_bboxes, gazelle_output, fail_indexes, all_ds_outputs, child_ids)
                assert len(trial_tracker) == len(gaze_scores)
                seg_score = np.zeros(len(gaze_label))
                for i in range(len(gaze_label)):
                    trial_ind = np.array(trial_tracker) == i
                    seg_score[i], j = np.max(gaze_scores[trial_ind]), np.argmax(gaze_scores[trial_ind])
                    if plotting:
                        images_in_trial = [overlay_images[t] for t, tr in enumerate(trial_ind) if tr]
                        images_to_plot.append(images_in_trial[j])
                gaze_pred = seg_score > 0.35
                acc = np.mean(gaze_pred == gaze_label)
                specificity = np.sum((gaze_pred == 0) & (gaze_label == 0)) / np.sum(gaze_label == 0)
                sensitivity = np.sum((gaze_pred == 1) & (gaze_label == 1)) / np.sum(gaze_label == 1)
                results.append((os.path.basename(video_file), acc, specificity, sensitivity))
                print("Accuracy:", acc)
                if plotting:
                    for i in range(len(gaze_label)):
                        plt.figure(figsize=(10,10))
                        plt.imshow(images_to_plot[i])
                        plt.axis('off')
                        plt.savefig(os.path.join("figures/gaze_with_tracking", os.path.basename(video_file).replace(".wmv", "_%d_%d.png" % (i, gaze_label[i]))), bbox_inches='tight', pad_inches=0)
            np.save("output/gaze_accs_tracking.npy", results) # save as it goes
        except IndexError:
            print("IndexError for video %s, skipping..." % video_file)
            continue
    return results

def visualize_gaze_in_whole_video(model, transform, video_file, face_threshold=0.5):
    import matplotlib.animation as animation

    images, _ = extract_frame_at_time_ranges(video_file, [[90.00, 120.00]]) # minute 5 to 10
    _, overlay_images = calculate_gaze_score_in_image(model, transform, images, face_threshold=face_threshold, cluster_child_pos=False)
    
    fig, ax = plt.subplots()
    img_display = ax.imshow(overlay_images[0], animated=True)
    ax.axis('off')
    
    def update(frame):
        img_display.set_array(overlay_images[frame])
        return [img_display]
    
    ani = animation.FuncAnimation(fig, update, frames=len(overlay_images), interval=33.33, blit=True)
    ani.save("output/%s_gaze_min5.mp4" % os.path.basename(video_file).replace(".wmv", ""), writer='ffmpeg', fps=30, dpi=200)

def plot_face_movement(output_dir, video_file):
    output_name = video_file.split("/")[-1].replace(".wmv", ".png")
    try:
        _, norm_bboxes, _, _ = load_model_output(output_dir, video_file)
    except FileNotFoundError:
        print("No output file found for %s" % video_file)
        return
    all_pos = []            
    for bboxes in norm_bboxes:
        bboxes, _ = rearrange_bbox_l2r(bboxes)
        if len(bboxes) == 3:
            pos = []
            for bbox in bboxes: 
                xmin, ymin, xmax, ymax = bbox
                bbox_center_x = (xmin + xmax) / 2
                bbox_center_y = (ymin + ymax) / 2
                pos.append([bbox_center_x, bbox_center_y])
            all_pos.append(np.vstack(pos))
    all_pos = np.array(all_pos) # [frame, person, 2]
    plt.figure()
    plt.scatter(all_pos[0, 0, 0], all_pos[0, 0, 1], c='r', marker="x")
    plt.scatter(all_pos[0, 1, 0], all_pos[0, 1, 1], c='g', marker="x")
    plt.scatter(all_pos[0, 2, 0], all_pos[0, 2, 1], c='b', marker="x")
    
    plt.plot(all_pos[:, 0, 0], all_pos[:, 0, 1], 'r-', label='parent', alpha=0.5)
    plt.plot(all_pos[:, 1, 0], all_pos[:, 1, 1], 'g-', label='child', alpha=0.5)
    plt.plot(all_pos[:, 2, 0], all_pos[:, 2, 1], 'b-', label='experimenter', alpha=0.5)
 
   
    plt.savefig(f"figures/face_movement/{output_name}", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze prediction from video")
    parser.add_argument("--plotting", action="store_true", help="Plot the gaze prediction results")
    parser.add_argument("--data_folder", type=str, default="/Datasets/.Autism_Videos/data/bids/", help="Path to the data folder")
    parser.add_argument("--session_key", type=str, default="ses-18mo", help="Session key to filter videos")
    parser.add_argument("--video_path", type=str, help="Path to specific video file")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to process")
    parser.add_argument("--output_ann_video_only", action="store_true", help="Output annotation video only")
    parser.add_argument("--cluster_child_pos", action="store_true", help="Cluster child position for gaze prediction")
    parser.add_argument("--saving_gaze_heatmap_only", action="store_true", help="Save gaze heatmap only")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index for testing")
    parser.add_argument("--plot_face_movement", action="store_true", help="Plot face movement")
    parser.add_argument("--recompute_gaze", action="store_true", help="Recompute gaze heatmap")
    parser.add_argument("--sample_start_index", type=int, default=0, help="Skip x videos")
    parser.add_argument("--gazelle_output_dir", type=str, default="/Datasets/.Autism_Videos/data/derivatives/gazelle/", help="Output directory for Gazelle model results")
    parser.add_argument("--deep_sort", action="store_true", help="Use Deep SORT for tracking")
    args = parser.parse_args()
    data_folder = args.data_folder


    # video_file = "/Datasets/.Autism_Videos/data/bids/sub-9943/ses-18mo/sub-9943_ses-18mo_measure-CSBS_video.wmv"
    # xlsx_file_path = "/Datasets/.Autism_Videos/data/bids/sub-9943/ses-18mo/sub-9943_ses-18mo_measure-CSBS_coding.xlsx"
    
    if args.deep_sort:
        deepsort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

    try:
        video_files = pickle.load(open("output/%s_video_files.pkl" % args.session_key, "rb"))
    except FileNotFoundError:
        video_files = extract_video_paths(data_folder, args.session_key)
        pickle.dump(video_files, open("output/%s_video_files.pkl" % args.session_key, "wb"))                          
                            
    if args.video_path is not None:
        video_list = [args.video_path]
    elif args.sample_index > -1:
        video_list = [video_files[args.sample_index]]
    else:
        video_list = video_files[args.sample_start_index:args.num_samples] # just a sample for testing
            
    if args.output_ann_video_only:
        model, transform = load_gaze_model()
        for video_file in video_list:
            print(video_file)
            visualize_gaze_in_whole_video(model, transform, video_file, face_threshold=0.5)
    elif args.saving_gaze_heatmap_only:
        model, transform = load_gaze_model()
        for video_file in video_list:
            gaze_label, begin_times, end_times = load_annotation_file(video_file)
            if gaze_label is not None:
                if (not check_output_exists(args.gazelle_output_dir, video_file)) or args.recompute_gaze:
                    print(f"computing {video_file}")
                    imgs, trial_tracker = extract_frame_at_time_ranges(video_file, list(zip(begin_times, end_times)))
                    images, norm_bboxes, output, fail_indexes = estimate_gaze_in_image(model, transform, imgs, batch_size=args.batch_size, face_threshold=0.5)
                    save_model_output(args.gazelle_output_dir, video_file, images, norm_bboxes, output, fail_indexes)
            else:
                print("Gaze label length is 0 for %s" % video_file)
                continue
    elif args.plot_face_movement:
        for video_file in video_list:
            plot_face_movement(args.gazelle_output_dir, video_file)
    elif args.deep_sort:
        model, transform = load_gaze_model()
        predict_gaze_from_video_with_tracking(video_list, model, transform, args.gazelle_output_dir, batch_size=args.batch_size, plotting=args.plotting)
    else:   
        model, transform = load_gaze_model()
        predict_gaze_from_video(video_list, model, transform, args.gazelle_output_dir, batch_size=args.batch_size, plotting=args.plotting)
                
            
# %%
