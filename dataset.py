"""PD Scalogram Dataset Loader.

- Input: Configurable size RGB images (64×64 default, 84×84 for standard benchmarks)
- Normalization: Auto-computed from dataset
- Supports both:
  1. Pre-split folders (train/val/test subfolders)
  2. Auto-split from a single folder
"""
import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


CLASS_MAP = {'corona': 0, 'hf_nopd': 1, 'surface': 2, 'void': 3}


class PDScalogramPreSplit:
    """Dataset loader for pre-split folders (train/val/test already separated).
    
    Expected folder structure:
        data_path/
            train/
                corona/
                surface/
                nopd/
            val/
                corona/
                surface/
                nopd/
            test/
                corona/
                surface/
                nopd/
    """
    
    def __init__(self, data_path, image_size=64):
        """
        Args:
            data_path: Path to dataset directory containing train/val/test subfolders
            image_size: Input image size (default: 64, use 84 for standard benchmarks)
        """
        self.data_path = os.path.abspath(data_path)
        self.image_size = image_size
        self.classes = sorted(CLASS_MAP.keys(), key=lambda c: CLASS_MAP[c])
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # File lists placeholders
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset (Pre-split): {self.data_path}')
        
        # 1. Scan pre-split folders
        self._scan_presplit_folders()
        
        # 2. Compute stats (ONLY on training data)
        self._compute_stats()
        
        # 3. Load images (apply normalization)
        self._load_images()
        
        self._shuffle_all()
    
    def _scan_presplit_folders(self):
        """Scan train/val/test subfolders and collect file paths."""
        splits = {
            'train': self.train_files,
            'val': self.val_files, 
            'test': self.test_files
        }
        
        for split_name, file_list in splits.items():
            split_path = os.path.join(self.data_path, split_name)
            if not os.path.exists(split_path):
                print(f"Warning: {split_name} folder not found at {split_path}")
                continue
            
            for class_name in CLASS_MAP:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    print(f"Warning: Class folder not found: {class_path}")
                    continue
                
                # Get all image files, excluding labeled versions
                files = sorted([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                               and 'labeled' not in f.lower()
                               and 'labeled:' not in f])
                
                label = CLASS_MAP[class_name]
                file_list.extend([(os.path.join(class_path, f), label) for f in files])
        
        print(f'Found: Train={len(self.train_files)}, Val={len(self.val_files)}, Test={len(self.test_files)}')
    
    def _compute_stats(self):
        """Compute per-channel mean and std using ONLY training data."""
        print('Computing mean/std on training set...')
        pixels = []
        
        for fpath, _ in self.train_files:
            img = Image.open(fpath).convert('RGB')  # RGB
            pixels.append(self._base_transform(img).numpy())
        
        if not pixels:
            print("Warning: No training data found for stats computation. Using default mean/std.")
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            all_imgs = np.stack(pixels)  # (N, 3, H, W)
            self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
            self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Final transform with normalization
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_images(self):
        """Load images using the pre-computed splits and normalization."""
        # Load Train
        for fpath, label in self.train_files:
            img = Image.open(fpath).convert('RGB')
            self.X_train.append(self.transform(img).numpy())
            self.y_train.append(label)
            
        # Load Val
        for fpath, label in self.val_files:
            img = Image.open(fpath).convert('RGB')
            self.X_val.append(self.transform(img).numpy())
            self.y_val.append(label)
            
        # Load Test
        for fpath, label in self.test_files:
            img = Image.open(fpath).convert('RGB')
            self.X_test.append(self.transform(img).numpy())
            self.y_test.append(label)
        
        # Convert to arrays
        self.X_train = np.array(self.X_train) if self.X_train else np.array([])
        self.y_train = np.array(self.y_train) if self.y_train else np.array([])
        self.X_val = np.array(self.X_val) if self.X_val else np.array([])
        self.y_val = np.array(self.y_val) if self.y_val else np.array([])
        self.X_test = np.array(self.X_test) if self.X_test else np.array([])
        self.y_test = np.array(self.y_test) if self.y_test else np.array([])
        
        print(f'Loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}')
    
    def _shuffle_all(self):
        """Shuffle all splits with fixed seed."""
        if len(self.X_train) > 0:
            idx = np.arange(len(self.X_train))
            np.random.default_rng(0).shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]
        
        if len(self.X_val) > 0:
            idx = np.arange(len(self.X_val))
            np.random.default_rng(1).shuffle(idx)
            self.X_val = self.X_val[idx]
            self.y_val = self.y_val[idx]
        
        if len(self.X_test) > 0:
            idx = np.arange(len(self.X_test))
            np.random.default_rng(2).shuffle(idx)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]


def load_dataset(data_path, image_size=64, val_per_class=60, test_per_class=60):
    """Auto-detect dataset structure and load appropriately.
    
    If data_path contains train/val/test subfolders, use PDScalogramPreSplit.
    Otherwise, use PDScalogram (auto-split).
    
    Args:
        data_path: Path to dataset
        image_size: Input image size
        val_per_class: (for auto-split only) Validation samples per class
        test_per_class: (for auto-split only) Test samples per class
    
    Returns:
        Dataset object (either PDScalogramPreSplit or PDScalogram)
    """
    # Check if pre-split structure exists
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    test_path = os.path.join(data_path, 'test')
    
    if os.path.isdir(train_path) and os.path.isdir(val_path) and os.path.isdir(test_path):
        print("Detected pre-split dataset structure (train/val/test folders)")
        return PDScalogramPreSplit(data_path, image_size=image_size)
    else:
        print("Using auto-split dataset structure")
        return PDScalogram(data_path, val_per_class=val_per_class, 
                          test_per_class=test_per_class, image_size=image_size)


class PDScalogram:
    """Dataset loader with auto-computed normalization (from training set only)."""
    
    def __init__(self, data_path, val_per_class=60, test_per_class=60, image_size=64):
        """
        Args:
            data_path: Path to dataset directory
            val_per_class: Samples reserved for validation per class
            test_per_class: Samples reserved for test per class
            image_size: Input image size (default: 64, use 84 for standard benchmarks)
        """
        self.data_path = os.path.abspath(data_path)
        self.val_per_class = val_per_class
        self.test_per_class = test_per_class
        self.image_size = image_size
        self.classes = sorted(CLASS_MAP.keys(), key=lambda c: CLASS_MAP[c])
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # File lists placeholders
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset: {self.data_path}')
        
        # 1. Prepare splits (identify files for train/val/test)
        self._prepare_splits()
        
        # 2. Compute stats (ONLY on training data)
        self._compute_stats()
        
        # 3. Load images (apply normalization)
        self._load_images()
        
        self._shuffle_all()
    
    def _prepare_splits(self):
        """Scan directories and split files into train/val/test lists."""
        # Find min class size
        class_sizes = {}
        for class_name in CLASS_MAP:
            path = os.path.join(self.data_path, class_name)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                        and 'labeled' not in f.lower()
                        and 'labeled:' not in f]
                class_sizes[class_name] = len(files)
            else:
                class_sizes[class_name] = 0
        
        if not class_sizes:
            raise ValueError(f"No data found in {self.data_path}")

        min_size = min(class_sizes.values())
        if min_size == 0:
            print("Warning: Found empty class or no images.")
            return {}, {}, {}

        val_size = min(self.val_per_class, min_size)
        test_size = min(self.test_per_class, min_size - val_size)
        eval_total = val_size + test_size
        
        print(f'Split: {val_size}/class for val, {test_size}/class for test, rest for train')
        
        for class_name in CLASS_MAP:
            path = os.path.join(self.data_path, class_name)
            if not os.path.exists(path):
                continue
                
            files = sorted([f for f in os.listdir(path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                           and 'labeled' not in f.lower()
                           and 'labeled:' not in f])
            random.Random(42).shuffle(files)
            files = files[:min_size]  # Balance classes
            
            # Split: val_size for val, test_size for test, rest for train
            val_files_class = files[:val_size]
            test_files_class = files[val_size:eval_total]
            train_files_class = files[eval_total:]
            
            # Store as (full_path, label) tuples
            label = CLASS_MAP[class_name]
            self.val_files.extend([(os.path.join(path, f), label) for f in val_files_class])
            self.test_files.extend([(os.path.join(path, f), label) for f in test_files_class])
            self.train_files.extend([(os.path.join(path, f), label) for f in train_files_class])

    def _compute_stats(self):
        """Compute per-channel mean and std using ONLY training data."""
        print('Computing mean/std on training set...')
        pixels = []
        
        for fpath, _ in self.train_files:
            img = Image.open(fpath).convert('RGB')
            pixels.append(self._base_transform(img).numpy())
        
        if not pixels:
            print("Warning: No training data found for stats computation. Using default mean/std.")
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            all_imgs = np.stack(pixels)  # (N, 3, H, W)
            self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
            self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Final transform with normalization
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_images(self):
        """Load images using the pre-computed splits and normalization."""
        # Load Train
        for fpath, label in self.train_files:
            img = Image.open(fpath).convert('RGB')
            self.X_train.append(self.transform(img).numpy())
            self.y_train.append(label)
            
        # Load Val
        for fpath, label in self.val_files:
            img = Image.open(fpath).convert('RGB')
            self.X_val.append(self.transform(img).numpy())
            self.y_val.append(label)
            
        # Load Test
        for fpath, label in self.test_files:
            img = Image.open(fpath).convert('RGB')
            self.X_test.append(self.transform(img).numpy())
            self.y_test.append(label)
        
        # Convert to arrays
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        print(f'Loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}')
    
    def _shuffle_all(self):
        """Shuffle all splits with fixed seed."""
        if len(self.X_train) > 0:
            idx = np.arange(len(self.X_train))
            np.random.default_rng(0).shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]
        
        if len(self.X_val) > 0:
            idx = np.arange(len(self.X_val))
            np.random.default_rng(1).shuffle(idx)
            self.X_val = self.X_val[idx]
            self.y_val = self.y_val[idx]
        
        if len(self.X_test) > 0:
            idx = np.arange(len(self.X_test))
            np.random.default_rng(2).shuffle(idx)
            self.X_test = self.X_test[idx]
            self.y_test = self.y_test[idx]
