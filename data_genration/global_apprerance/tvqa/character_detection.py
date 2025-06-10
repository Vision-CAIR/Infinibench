import os
from ultralytics import YOLO
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from decord import VideoReader, cpu
import logging
import random

def load_n_video_frames(video_path, number_of_frames=8):
    vr=VideoReader(video_path,num_threads=1, ctx=cpu(0))
    total_frame_num = len(vr)
    sampling_rate = max (int(total_frame_num/number_of_frames),1)
    sample_frame_idx = [i for i in range(0, total_frame_num, sampling_rate)]
    sample_frame_idx = sample_frame_idx[:number_of_frames]
    img_array = vr.get_batch(sample_frame_idx).asnumpy()
    frames_time=[round(i/vr.get_avg_fps(),1) for i in sample_frame_idx]
    # Decord frames are in RGB format
    # Convert to PIL images 
    # pil_images = [Image.fromarray(frame) for frame in img_array]
    return img_array ,frames_time


def load_n_frames_one_by_one(video_path, number_of_frames=8):
    vr=VideoReader(video_path,num_threads=1, ctx=cpu(0))
    total_frame_num = len(vr)
    sampling_rate = max (int(total_frame_num/number_of_frames),1)
    
    # Generate initial list of frame indices to sample
    candidate_frame_idx = list(range(0, total_frame_num, sampling_rate))
    # Ensure there are enough candidates
    while len(candidate_frame_idx) < number_of_frames:
        candidate_frame_idx.append(total_frame_num - 1)
    
    
    img_list = []
    logging.warning(f"Loading all non corrupted samples frames")
    for idx in candidate_frame_idx:
        try:
            # Attempt to load the frame
            frame = vr[idx].asnumpy()
            img_list.append((idx,frame))
            # Break if we've collected enough frames
            if len(img_list) >= number_of_frames:
                break
        except Exception as e:
            # Log the error and skip the corrupted frame
            logging.warning(f"Skipping corrupted frame at index {idx}: {e}")
            continue
    
    logging.warning(f"Loaded {len(img_list)} frames out of {number_of_frames}")
    logging.warning("Attempting to load additional random frames to reach the desired number of frames")
    
    # If not enough frames are loaded, attempt to load additional frames
    if len(img_list) < number_of_frames:
        additional_idx = list(set(range(total_frame_num)) - set(candidate_frame_idx))
        random.shuffle(additional_idx) # Shuffle the indices to load random frames
        for idx in additional_idx:
            if len(img_list) >= number_of_frames:
                break
            try:
                frame = vr[idx].asnumpy()
                img_list.append((idx,frame))
                logging.warning(f"Loaded frame at index {idx}")
            except Exception as e:
                logging.warning(f"Skipping corrupted frame at index {idx}: {e}")
                continue
    if len(img_list) == number_of_frames:
        logging.warning("Successfully loaded enough frames")
    else:
        logging.warning("Failed to load additional frames, padding with the last valid frame")
    # If still not enough frames, pad with the last valid frame or zeros
    while len(img_list) < number_of_frames:
        if img_list:
            img_list.append(img_list[-1])
        else:
            # If no frames were loaded successfully, return empty arrays
            logging.warning("No valid frames were loaded")
            return np.array([]), []
    logging.warning("Successfully loaded frames")
    # Sort the frames by their original index
    img_list.sort(key=lambda x: x[0])
    frames_time=[round(i[0]/vr.get_avg_fps(),1) for i in img_list]
    img_array = np.array([x[1] for x in img_list])
    # Decord frames are in RGB format
    # Convert to PIL images 
    # pil_images = [Image.fromarray(frame) for frame in img_array]
    return img_array ,frames_time

def load_video_frames(video_path, fps=1):
    if fps==0:
        try:
            return load_n_video_frames(video_path, number_of_frames=8)
        except Exception as e:
            logging.warning(f"Video contains corrupted frames, try to load them one by one to exclude corrupted frames: {e}")
            return load_n_frames_one_by_one(video_path, number_of_frames=8)
    try:
        vr = VideoReader(video_path,num_threads=1, ctx=cpu(0))
        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()
        sample_frame_idx = [i for i in range(0, total_frame_num, int(video_fps/fps))]  
        img_array = vr.get_batch(sample_frame_idx).asnumpy()
        frames_time=[round(i/video_fps,1) for i in sample_frame_idx]
        # Decord frames are in RGB format
        # Convert to PIL images 
        # pil_images = [Image.fromarray(frame) for frame in img_array]
        return img_array ,frames_time
    except Exception as e:
        logging.warning(f"Video contains corrupted frames, try to load them one by one to exclude corrupted frames: {e}")
        return load_video_frames_one_by_one(video_path,fps=fps)

def load_video_frames_one_by_one(video_path, fps):
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    total_frame_num = len(vr)
    video_fps = vr.get_avg_fps()
    sample_gap = int(video_fps / fps)
    
    # Generate initial list of frame indices to sample
    candidate_frame_idx = list(range(0, total_frame_num, sample_gap))
    img_list = []
    for idx in candidate_frame_idx:
        try:
            # Attempt to load the frame
            frame = vr[idx].asnumpy()
            img_list.append(frame)
        except Exception as e:
            # Log the error and skip the corrupted frame
            logging.warning(f"Skipping corrupted frame at index {idx}: {e}")
            continue
    img_array=np.array(img_list)
    frames_time=[round(i/video_fps,1) for i in candidate_frame_idx]
    # Decord frames are in RGB format
    # Convert to PIL images 
    # pil_images = [Image.fromarray(frame) for frame in img_array]
    return img_array ,frames_time




def is_inside(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2
    return box1_x1 >= box2_x1 and box1_y1 >= box2_y1 and box1_x2 <= box2_x2 and box1_y2 <= box2_y2

def process_frame( frame,frame_time, results, faces,save_path):
    character_bbox = {}
    # Match faces to characters
    for face in faces:
        face_embedding = torch.tensor(face.normed_embedding).to(f"cuda:{device_id}")
        similarities = torch.cosine_similarity(face_embedding.unsqueeze(0), show_embeddings_tensor, dim=1)
        arg_max = torch.argmax(similarities)
        similarity = round(similarities[arg_max].item(), 2)
        if similarity >= 0.2:
            character_name = characters_names[arg_max]
            character_bbox[character_name] = (face.bbox, similarity)

    # Annotate frame
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for character_name, face_data in character_bbox.items():
                face_box, similarity = face_data
                if is_inside(face_box, box):
                    # crop character 
                    cropped_character = frame[y1:y2, x1:x2]
                    # save character image 
                    os.makedirs(os.path.join(save_path, character_name), exist_ok=True)
                    cv2.imwrite(os.path.join(save_path, character_name, f"{frame_time}.jpg"), cropped_character)
                    # cv2.putText(frame, f"{character_name}_{similarity}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

def detect_characters(video_path,save_path,fps=1):
    # cap = cv2.VideoCapture(video_path)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # Read all frames
    rgb_frames, frames_time = load_video_frames(video_path, fps=fps)
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames]
    # Batch process YOLO detections
    batch_size=128
    all_detections_results=[]
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        all_detections_results.extend(model(batch_frames, classes=0))
    # indecies of frames that have persons 
    frames_with_persons = []
    frame_num_with_persons=0
    indices_with_persons_mapping={}
    for idx, result in enumerate(all_detections_results):
        if len(result):
            frames_with_persons.append(rgb_frames[idx])
            indices_with_persons_mapping[idx]=frame_num_with_persons
            frame_num_with_persons+=1
        
    # Run face detection in parallel
    max_cpus=os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_cpus) as executor:
        futures = {executor.submit(app.get, frame): idx for idx, frame in enumerate(frames_with_persons)}
        all_faces_results = [None] * len(frames_with_persons)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                all_faces_results[idx] = future.result()
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")

    # Process and write frames
    for idx, frame in enumerate(frames):
        if indices_with_persons_mapping.get(idx,False):
            face_index=indices_with_persons_mapping[idx]
            frame_time=frames_time[idx]
            frame = process_frame(frame,frame_time, all_detections_results[idx], all_faces_results[face_index],save_path)
        # video_writer.write(frame)

    # video_writer.release()



parser = argparse.ArgumentParser(description="character_detection")
parser.add_argument("--show", default="friends", help="Name of the show")
parser.add_argument("--season", default="season_1", help="Season of the show")
parser.add_argument("--videos_path", default="", help="Path to the videos directory")
parser.add_argument("--output_dir", default="characters", help="Output directory for character images")
args = parser.parse_args()
show = args.show
season = args.season

device_id = 0
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'], 
                   provider_options=[{"device_id": str(device_id)}])
app.prepare(ctx_id=device_id, det_size=(640, 640))

# Load embeddings
with open(os.path.join("shows_characters_embs", f"{show}.pkl"), 'rb') as f:
    show_embeddings = pickle.load(f)
show_embeddings_tensor = torch.tensor(list(show_embeddings.values())).to(f"cuda:{device_id}")
characters_names = list(show_embeddings.keys())

model = YOLO("yolov8l.pt")


# Process all videos
videos_path = args.videos_path
# for season in os.listdir(os.path.join(videos_path, show)):
fps=0.5
for episode in os.listdir(os.path.join(videos_path, show, season)):
    video_path = os.path.join(videos_path, show, season, episode)
    print(f"Processing: {video_path}")
    save_path=os.path.join(args.output_dir, show, season, episode).replace(".mp4", "")
    os.makedirs(save_path, exist_ok=True)
    start_time = time.time()
    detect_characters(video_path,save_path,fps=fps)
    print(f"Time taken: {time.time() - start_time:.2f}s")
