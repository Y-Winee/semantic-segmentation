import os
import cv2

import torch
import torch.nn as nn

from torchvision import models
from torchvision.transforms import v2 as T

def extract_semantic_frames(input_path, output_dir, similarity_threshold=0.85, skip_frames=10):
    """
    Extracts unique frames from a video using ResNet50 semantic embeddings.
    
    Parameters:
    - input_path: Path to input video.
    - output_dir: Destination folder for extracted frames.
    - similarity_threshold: Vectors with a cosine similarity below this score 
                            are considered "new scenes" and extracted.
    - skip_frames: Evaluates only every Nth frame to drastically speed up processing.
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    weights = models.ResNet50_Weights.DEFAULT
    resNet = models.resnet50(weights=weights)

    preprocess = T.Compose([
        T.ToImage(),                             
        T.ToDtype(torch.float32, scale=True), 
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resNet.fc = nn.Identity()
    resNet.to(device)
    resNet.eval()

    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_embeddings(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            return resNet(input_tensor)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Video is empty.")
        return

    frame_count = 0
    saved_count = 1

    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:05d}.jpg"), prev_frame)
    prev_embedding = get_embeddings(prev_frame)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % skip_frames != 0:
            continue

        curr_embedding = get_embeddings(current_frame)
        similarity = cos_sim(prev_embedding, curr_embedding).item()

        if similarity < similarity_threshold:
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:05d}.jpg"), current_frame)
            saved_count += 1

            prev_embedding = curr_embedding

    cap.release()
    print(f"\nExtraction complete!")
    print(f"Total frames scanned: {frame_count}")
    print(f"Unique scenes saved: {saved_count}")


if __name__ == "__main__":
    VIDEO_FILE = "okinawa-trimmed.mp4"
    OUTPUT_FOLDER = "semantic_dataset"
    
    extract_semantic_frames(VIDEO_FILE, OUTPUT_FOLDER, similarity_threshold=0.97, skip_frames=15)