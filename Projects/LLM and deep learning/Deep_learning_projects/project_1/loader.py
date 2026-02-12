from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

class PNGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []
        self.label_to_idx = {}
        
        for idx, label in enumerate(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                self.label_to_idx[label] = idx
                for file in os.listdir(label_path):
                    if file.endswith('.png'):
                        self.files.append(os.path.join(label_path, file))
                        self.labels.append(idx)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Values of means and std were computed in older version and pasted here.
def load_png_images(root_dir, transform=[transforms.ToTensor(), transforms.Normalize(mean=[0.4788952171802521, 0.4722793698310852, 0.43047481775283813], std=[0.24205632507801056, 0.2382805347442627, 0.25874853134155273])], batch_size=32, shuffle=True, num_workers=2):
    transform = transforms.Compose(transform)
    dataset = PNGDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, len(dataset.label_to_idx)