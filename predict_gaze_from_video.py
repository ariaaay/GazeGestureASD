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


def visualize_all(pil_image, heatmaps, bboxes, inout_scores, child_pos, inout_thresh=0.5):
    # bboxes, l2r_order = rearrange_bbox_l2r(bboxes)
    distances = calculate_bbox_distance_to_child_pos(child_pos, bboxes)
    child_ind = np.argmin(distances)
    # heatmaps = heatmaps[l2r_order]
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

# def collate_fn(batch):
#     print("Collating batch of size:", len(batch))
#     print(len(batch[0]["inout"]))
#     print(len(batch[0]["heatmap"]))
#     print(len(batch[1]["inout"]))
#     print(len(batch[1]["heatmap"]))
#     collated = {}
#     for key in batch[0].keys():
#         values = [item[key] for item in batch]
#         try:
#             collated[key] = default_collate(values)
#         except RuntimeError:
#             # fallback to list if stacking fails (e.g., variable lengths)
#             print(key, "stacking failed, fallback to list")
#             collated[key] = values
#     return collated



def visualize_gaze_in_image(model, transform, images, face_threshold=0.5, batch_size=128):
    gaze_scores = []
    overlay_images = []
    width, height = images[0].size
    print("Number of images:", len(images))
    print("Image size:", width, height)
    
    norm_bboxes = []
    img_tensors = []
    fail_indexes = []
    for i, image in enumerate(images):
        resp = RetinaFace.detect_faces(np.array(image), threshold=face_threshold)
        bboxes = [resp[key]['facial_area'] for key in resp.keys()]
        if bboxes is None or len(bboxes) == 0:
            print("No face detected in image %d" % i)
            fail_indexes.append(i)
        else:
            img_tensors.append(transform(image).unsqueeze(0).to(device))
            norm_bboxes.append([np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes])
        
    child_pos = cluster_bbox_for_child_pos(norm_bboxes)
    
    input = {
        "images": torch.cat(img_tensors, 0), # [num_images, 3, 448, 448]
        "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
    }    
    
    # to avoid OOM, we split the input into microbatches
    total_size = len(images)
    if total_size > batch_size:
        output_heatmap, output_inout = [], []
        for i in tqdm(range(0, total_size, batch_size)):
            microbatch = {
                "images": input["images"][i:i+batch_size],
                "bboxes": input["bboxes"][i:i+batch_size]}
            with torch.no_grad(): 
                batch_output = model(microbatch)
            # print(len(batch_output["heatmap"]))
            # print(len(batch_output["heatmap"][0]))
            # print(batch_output["heatmap"][0][0])
            output_heatmap += batch_output["heatmap"]
            # print(len(output_heatmap))
            output_inout += batch_output["inout"]

        # output = collate_fn(outputs)
        output = {
            "heatmap": output_heatmap,
            "inout": output_inout
        }
    else:
        with torch.no_grad():
            output = model(input)
    
    k = 0
    for i, image in enumerate(images):
        if i in fail_indexes:
            overlay_images.append(image)
            gaze_scores.append(-1)
            k+=1
        else:
            vis, score = visualize_all(image, output['heatmap'][i-k], norm_bboxes[i-k], output['inout'][i-k] if output['inout'][i-k] is not None else None, child_pos, inout_thresh=0.5)
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
            raise ValueError("Requested time exceeds video duration")
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

def predict_gaze_from_video(video_list, model, transform, plotting):
    results = []
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
        imgs, trial_tracker = extract_frame_at_time_ranges(video_file, list(zip(begin_times, end_times)))
        if plotting:
            images_to_plot = []
        if label_len > 0:
            gaze_scores, overlay_images = visualize_gaze_in_image(model, transform, imgs)
            # print(gaze_scores)
            # print(gaze_label)
            # print(trial_tracker)
            assert len(trial_tracker) == len(gaze_scores)
            seg_score = np.zeros(label_len)
            for i in range(label_len):
                trial_ind = trial_tracker == i
                seg_score[i], j = np.max(gaze_scores[trial_ind]), np.argmax(gaze_scores[trial_ind])
                images_to_plot.append(np.array(overlay_images)[trial_ind][j])
            gaze_pred = seg_score > 0.2
            acc = np.mean(gaze_pred == gaze_label)
            specificity = np.sum((gaze_pred == 0) & (gaze_label == 0)) / np.sum(gaze_label == 0)
            sensitivity = np.sum((gaze_pred == 1) & (gaze_label == 1)) / np.sum(gaze_label == 1)
            results.append((os.path.basename(video_file), acc, specificity, sensitivity))
            print("Accuracy:", acc)
            if plotting:
                for i in range(label_len):
                    plt.figure(figsize=(10,10))
                    plt.imshow(images_to_plot[i])
                    plt.axis('off')
                    plt.savefig(os.path.join("figures", os.path.basename(video_file).replace(".wmv", "_%d.png" % i)), bbox_inches='tight', pad_inches=0)
    np.save("output/gaze_accs.npy", results)
    return results


def visualize_gaze_in_whole_video(model, transform, video_file, face_threshold=0.5):
    import matplotlib.animation as animation

    images = extract_all_frames(video_file)
    _, overlay_images = visualize_gaze_in_image(model, transform, images, face_threshold=face_threshold)
    
    fig, ax = plt.subplots()
    img_display = ax.imshow(overlay_images[0], animated=True)
    ax.axis('off')
    
    def update(frame):
        img_display.set_array(overlay_images[frame])
        return [img_display]
    
    ani = animation.FuncAnimation(fig, update, frames=len(overlay_images), interval=100, blit=True)
    ani.save("output/%s_gaze.mp4" % os.path.basename(video_file).replace(".wmv", ""), writer='ffmpeg', fps=30)

# video_file = "/Datasets/.Autism_Videos/data/bids/sub-9943/ses-18mo/sub-9943_ses-18mo_measure-CSBS_video.wmv"
# xlsx_file_path = "/Datasets/.Autism_Videos/data/bids/sub-9943/ses-18mo/sub-9943_ses-18mo_measure-CSBS_coding.xlsx"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze prediction from video")
    parser.add_argument("--plotting", action="store_true", help="Plot the gaze prediction results")
    parser.add_argument("--data_folder", type=str, default="/Datasets/.Autism_Videos/data/bids/", help="Path to the data folder")
    parser.add_argument("--session_key", type=str, default="ses-18mo", help="Session key to filter videos")
    parser.add_argument("--video_path", type=str, help="Path to specific video file")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to process")
    parser.add_argument("--output_ann_video_only", action="store_true", help="Output annotation video only")
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
    
    if args.output_ann_video_only:
        for video_file in video_list:
            print(video_file)
            visualize_gaze_in_whole_video(model, transform, video_file, face_threshold=0.5)
    else:
        predict_gaze_from_video(video_list, model, transform, plotting=args.plotting)
                
            