import numpy as np
import os
import random
import torch
import torchaudio
from noisereduce import reduce_noise
from pathlib import Path
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

raw_path = os.path.join("data", "train", "train")
audio_root = os.path.join(raw_path, "audio")
test_list_path = os.path.join(raw_path, 'testing_list.txt')
val_list_path = os.path.join(raw_path, 'validation_list.txt')
output_root = os.path.join("data", "preprocessed")


def preprocess_and_save_audio_in_tensors(denoise=False):
    counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_transform = MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=64
    )
    toDB = AmplitudeToDB()
    if denoise:
        output_path = os.path.join(output_root, 'denoised')
    else:
        output_path = os.path.join(output_root, 'standard')

    with open(test_list_path, 'r') as f:
        test_files = set(line.strip() for line in f if line.strip())
    with open(val_list_path, 'r') as f:
        val_files = set(line.strip() for line in f if line.strip())

    target_length = 16000
    for subdir, _, files in os.walk(audio_root):
        if '_background_noise_' in subdir:
            continue

        for file in files:
            if not file.endswith(".wav"):
                continue

            full_path = os.path.join(subdir, file)
            rel_path = os.path.relpath(full_path, audio_root).replace("\\", "/")

            if rel_path in test_files:
                split = 'test'
            elif rel_path in val_files:
                split = 'validation'
            else:
                split = 'train'

            label = rel_path.split('/')[0]
            filename = os.path.splitext(os.path.basename(file))[0]

            waveform, sr = torchaudio.load(full_path, num_frames=target_length, backend='soundfile')

            if sr != target_length:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            current_length = waveform.size(1)

            if current_length < target_length:
                # Padding too short samples
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            if denoise:
                waveform = reduce_noise(waveform, sr, use_torch=True, device=device)
                waveform = torch.tensor(waveform)
                waveform = torch.nan_to_num(waveform, nan=0)

            waveform.to(device)
            mel = mel_transform(waveform)
            mel = toDB(mel)

            raw_out_path = Path(output_path) / "raw" / split / label / f"{filename}.pt"
            mel_out_path = Path(output_path) / "mel" / split / label / f"{filename}.pt"

            raw_out_path.parent.mkdir(parents=True, exist_ok=True)
            mel_out_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(waveform, raw_out_path)
            torch.save(mel, mel_out_path)

            counter += 1

            if counter == 10000:
                print("10 000 files proccessed")
                counter = 0

            # print(f"Saved: {rel_path} → {split}")


def generate_dataset_with_optional_augmented_sample(
        split_ratios=(0.6, 0.2, 0.2),
        denoise=False
):
    chunk_size = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_transform = MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=64
    )
    toDB = AmplitudeToDB()
    shifts = [0, chunk_size // 3, 2 * chunk_size // 3]
    if denoise:
        out_base = os.path.join(output_root, "noise", "denoised")
    else:
        out_base = os.path.join(output_root, "noise", 'standard')

    def pad_or_crop(waveform, start, size):
        chunk = waveform[:, start:start + size]
        if chunk.shape[1] < size:
            padding = size - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        return chunk

    random.seed(42)
    torch.manual_seed(42)

    all_chunks = []
    noise_dir = os.path.join(audio_root, "_background_noise_")
    #
    for file in Path(noise_dir).glob("*.wav"):
        waveform, _ = torchaudio.load(file)
        waveform = waveform.mean(dim=0, keepdim=True)
        length = waveform.shape[1]

        for shift in shifts:
            for start in range(0, length - shift, chunk_size):
                chunk = pad_or_crop(waveform, start + shift, chunk_size)
                all_chunks.append(chunk)
    print(len(all_chunks))
    # 25% new observation from mixing
    for _ in range(len(all_chunks) // 4):
        c1, c2 = random.sample(all_chunks, 2)
        alpha = random.uniform(0.3, 0.7)
        mixed = (alpha * c1 + (1 - alpha) * c2).clamp(-1.0, 1.0)
        all_chunks.append(mixed)

    print(len(all_chunks))
    # 15% of adding new observation with gausian noise with 0.1 std
    for chunk in all_chunks:
        if random.random() < 0.15:
            noise = torch.randn_like(chunk) * 0.1
            noised = mixed + noise
            all_chunks.append((noised).clamp(-1.0, 1.0))

    indices = list(range(len(all_chunks)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(split_ratios[0] * n_total)
    n_val = int(split_ratios[1] * n_total)

    split_map = {}
    for i, idx in enumerate(indices):
        if i < n_train:
            split_map[idx] = "train"
        elif i < n_train + n_val:
            split_map[idx] = "validation"
        else:
            split_map[idx] = "test"

    for idx, waveform in enumerate(all_chunks):
        split = split_map[idx]
        if denoise:
            waveform = reduce_noise(waveform, 16000, use_torch=True, device="cuda")
            waveform = torch.tensor(waveform)
            waveform = torch.nan_to_num(waveform, nan=0)

        # mel = torch.tensor(waveform)
        waveform.to(device)
        mel = mel_transform(waveform)
        mel = toDB(mel)

        raw_dir = os.path.join(out_base, "raw", split)
        mel_dir = os.path.join(out_base, "mel", split)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(mel_dir, exist_ok=True)
        torch.save(waveform, os.path.join(raw_dir, f"waveform_{idx:05d}.pt"))
        torch.save(mel, os.path.join(mel_dir, f"mel_{idx:05d}.pt"))

    print(len(all_chunks))


class TorchTensorFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.class_to_idx = {}
        self._prepare_file_list()

    def _prepare_file_list(self):
        for class_dir in sorted(self.root_dir.glob("*")):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            for file in class_dir.glob("*.pt"):
                self.samples.append((file, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path, weights_only=False)

        return tensor, label
