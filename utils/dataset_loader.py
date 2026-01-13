# # FILE: utils/dataset_loader.py

# import os
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split

# class UCF50Dataset(Dataset):
#     def __init__(self, video_paths, labels, class_to_idx, sequence_length=16, img_size=112, transform=None):
#         """
#         Args:
#             video_paths (list): List of file paths to videos.
#             labels (list): List of string labels corresponding to videos.
#             class_to_idx (dict): Mapping from label string to integer index.
#             sequence_length (int): Number of frames to extract (T).
#             img_size (int): Height/Width of frames (H, W).
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.video_paths = video_paths
#         self.labels = labels
#         self.class_to_idx = class_to_idx
#         self.sequence_length = sequence_length
#         self.img_size = img_size
#         self.transform = transform

#     def __len__(self):
#         return len(self.video_paths)

#     def _extract_frames(self, video_path):
#         """
#         Reads video, extracts exactly 'sequence_length' frames uniformly.
#         """
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         try:
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
#             # Safety check: if video is unreadable or empty
#             if total_frames == 0:
#                 # Return zeros if video is broken
#                 return np.zeros((self.sequence_length, self.img_size, self.img_size, 3), dtype=np.uint8)

#             # Uniform sampling indices
#             if total_frames < self.sequence_length:
#                 # If video is too short, take all available and pad later
#                 indices = list(range(total_frames))
#             else:
#                 # If long enough, take evenly spaced frames
#                 indices = np.linspace(0, total_frames - 1, self.sequence_length).astype(int)

#             for i in range(total_frames):
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 if i in indices:
#                     # Resize and Convert BGR (OpenCV) to RGB
#                     frame = cv2.resize(frame, (self.img_size, self.img_size))
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     frames.append(frame)
                    
#                     if len(frames) == self.sequence_length:
#                         break
#         finally:
#             cap.release()

#         # Convert to numpy array
#         frames = np.array(frames)

#         # Handle padding if we didn't get enough frames
#         if len(frames) < self.sequence_length:
#             padding = np.zeros((self.sequence_length - len(frames), self.img_size, self.img_size, 3), dtype=np.uint8)
#             if len(frames) > 0:
#                 frames = np.concatenate((frames, padding), axis=0)
#             else:
#                 frames = padding

#         return frames

#     def __getitem__(self, idx):
#         video_path = self.video_paths[idx]
#         label_name = self.labels[idx]
        
#         # 1. Load raw frames: Shape (T, H, W, C)
#         frames = self._extract_frames(video_path)
        
#         # 2. Preprocess: Normalize to [0, 1]
#         frames = frames / 255.0
        
#         # 3. Convert to Tensor
#         # PyTorch 3D CNNs expect: (Channel, Depth/Time, Height, Width) -> (C, T, H, W)
#         frames_tensor = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        
#         # 4. Get Label Index
#         label_idx = self.class_to_idx[label_name]
        
#         return frames_tensor, label_idx

# def split_dataset(dataset_root, test_size=0.2, random_state=42):
#     """
#     Scans the dataset folder and creates Train/Test splits.
#     Returns: (train_paths, train_labels), (test_paths, test_labels), class_to_idx
#     """
#     classes = sorted(os.listdir(dataset_root))
#     # Filter out hidden files or non-folders
#     classes = [c for c in classes if os.path.isdir(os.path.join(dataset_root, c))]
    
#     class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
#     all_paths = []
#     all_labels = []

#     print(f"Scanning {dataset_root}...")
#     for cls in classes:
#         cls_path = os.path.join(dataset_root, cls)
#         # Handle variations in file extensions
#         videos = [v for v in os.listdir(cls_path) if v.endswith(('.avi', '.mp4'))]
        
#         for vid in videos:
#             all_paths.append(os.path.join(cls_path, vid))
#             all_labels.append(cls)
            
#     print(f"Found {len(all_paths)} videos in {len(classes)} classes.")
    
#     # Stratified Split
#     train_paths, test_paths, train_labels, test_labels = train_test_split(
#         all_paths, all_labels, test_size=test_size, stratify=all_labels, random_state=random_state
#     )
    
#     return (train_paths, train_labels), (test_paths, test_labels), class_to_idx






































## update 1 
# optimized dataset loder 

# FILE: utils/dataset_loader.py

import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class UCFFastDataset(Dataset):
    def __init__(self, file_paths, labels, class_to_idx):
        self.file_paths = file_paths
        self.labels = labels
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Load the ByteTensor (integers 0-255)
        # Shape: (C, T, H, W)
        frames_int = torch.load(self.file_paths[idx])
        
        # 2. Convert to Float and Normalize (0.0 - 1.0)
        # This takes microseconds
        frames_float = frames_int.float() / 255.0
        
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]
        
        return frames_float, label_idx

def split_fast_dataset(dataset_root, test_size=0.2):
    classes = sorted(os.listdir(dataset_root))
    classes = [c for c in classes if os.path.isdir(os.path.join(dataset_root, c))]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    all_paths = []
    all_labels = []

    print(f"Scanning {dataset_root}...")
    for cls in classes:
        cls_path = os.path.join(dataset_root, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith('.pt')]
        for f in files:
            all_paths.append(os.path.join(cls_path, f))
            all_labels.append(cls)
            
    print(f"Found {len(all_paths)} processed tensors.")
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, stratify=all_labels, random_state=42
    )
    
    return (train_paths, train_labels), (test_paths, test_labels), class_to_idx