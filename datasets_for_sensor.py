import os
import re
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PAPER_HYPERPARAMS = {
    'seq_len': 300,
    'fps': 10,
    'img_size': 224,
    'batch_size': 32,
    'epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
}

RLDD_CLASS_NAMES = ['active', 'fatigue']
NTHU_CLASS_NAMES = ['notdrowsy', 'drowsy']


def _load_rldd_from_txt(archive_root, split):
    txt_path = os.path.join(archive_root, f'{split}.txt')
    samples = []
    if not os.path.exists(txt_path):
        return samples
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rel_path, label = parts[0], int(parts[-1])
                full_path = os.path.join(archive_root, rel_path.replace('/', os.sep))
                if os.path.exists(full_path):
                    samples.append((full_path, label))
    return samples


def _group_rldd_sequences(archive_root, split):
    sequences = []
    for class_idx, class_name in enumerate(RLDD_CLASS_NAMES):
        class_dir = os.path.join(archive_root, split, class_name)
        if not os.path.isdir(class_dir):
            continue
        groups = defaultdict(list)
        for fn in os.listdir(class_dir):
            if not fn.endswith('.jpg'):
                continue
            m = re.match(r'(img_([a-z])_|image_)(\d+)\.jpg', fn)
            if m:
                key = f'img_{m.group(2)}' if m.group(2) else 'image'
                groups[key].append((int(m.group(3)), os.path.join(class_dir, fn)))
        for subject_id, frames in groups.items():
            frames.sort(key=lambda x: x[0])
            if len(frames) >= 30:
                sequences.append({
                    'subject': f'{split}_{class_name}_{subject_id}',
                    'frames': [f[1] for f in frames],
                    'label': class_idx,
                })
    return sequences


def _group_nthu_sequences(nthu_root):
    sequences = []
    for class_idx, class_name in enumerate(NTHU_CLASS_NAMES):
        class_dir = os.path.join(nthu_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        groups = defaultdict(list)
        for fn in os.listdir(class_dir):
            if not fn.endswith('.jpg'):
                continue
            m = re.match(r'(.+)_(\d+)_(drowsy|notdrowsy)\.jpg$', fn)
            if m:
                groups[m.group(1)].append((int(m.group(2)), os.path.join(class_dir, fn)))
        for seq_id, frames in groups.items():
            frames.sort(key=lambda x: x[0])
            if len(frames) >= 30:
                sequences.append({
                    'subject': seq_id,
                    'frames': [f[1] for f in frames],
                    'label': class_idx,
                })
    return sequences


def _split_nthu_sequences(sequences, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    subjects = sorted(set(s['subject'].split('_')[0] for s in sequences))
    rng.shuffle(subjects)
    n_train = max(1, int(len(subjects) * train_ratio))
    n_val = max(1, int(len(subjects) * val_ratio))
    train_subs = set(subjects[:n_train])
    val_subs = set(subjects[n_train:n_train + n_val])
    splits = {'train': [], 'val': [], 'test': []}
    for seq in sequences:
        subj = seq['subject'].split('_')[0]
        if subj in train_subs:
            splits['train'].append(seq)
        elif subj in val_subs:
            splits['val'].append(seq)
        else:
            splits['test'].append(seq)
    return splits


class FatigueDataset(Dataset):

    def __init__(self, data_root='.', dataset_name='RLDD', split='train',
                 seq_len=None, img_size=None, use_sequence_grouping=True):
        hp = PAPER_HYPERPARAMS
        self.seq_len = seq_len if seq_len is not None else hp['seq_len']
        self.img_size = img_size if img_size is not None else hp['img_size']
        self.fps = hp['fps']

        if dataset_name == 'RLDD':
            archive_root = os.path.join(data_root, 'archive')
            self.num_classes = 2
            self.class_names = RLDD_CLASS_NAMES
            if use_sequence_grouping:
                self.sequences = _group_rldd_sequences(archive_root, split)
                self.frame_samples = None
            else:
                self.sequences = None
                self.frame_samples = _load_rldd_from_txt(archive_root, split)
                self.archive_root = archive_root
        else:
            all_seqs = _group_nthu_sequences(os.path.join(data_root, 'NTHU-DDD'))
            self.num_classes = 2
            self.class_names = NTHU_CLASS_NAMES
            self.sequences = _split_nthu_sequences(all_seqs).get(split, [])
            self.frame_samples = None

    def __len__(self):
        return len(self.sequences) if self.sequences is not None else len(self.frame_samples)

    def _load_image(self, path):
        try:
            with open(path, 'rb') as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except OSError:
            img = None
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _sample_frames(self, paths):
        n = len(paths)
        if n >= self.seq_len:
            idx = np.linspace(0, n - 1, self.seq_len, dtype=int)
        else:
            idx = np.array([i % n for i in range(self.seq_len)])
        frames = np.stack([self._load_image(paths[i]) for i in idx], axis=0)
        return torch.FloatTensor(frames.transpose(3, 0, 1, 2))

    def __getitem__(self, idx):
        if self.sequences is not None:
            seq = self.sequences[idx]
            label = seq['label']
            return self._sample_frames(seq['frames']), label, float(label)

        path, label = self.frame_samples[idx]
        return self._sample_frames([path]), label, float(label)
