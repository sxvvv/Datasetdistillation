import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import cv2
from Restormer import Restormer
from PIL import Image
from sklearn.metrics import pairwise_distances
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from image_utils import random_augmentation, crop_img
from transformers import ViTModel, ViTImageProcessor, CLIPModel, CLIPProcessor
import warnings
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
# Ignore warnings
warnings.filterwarnings("ignore")

# ===============================
# Dataset Definition
# ===============================
class ImageDataset(Dataset):
    def __init__(self, args, is_test=False):
        super().__init__()
        self.args = args
        self.data_ids = []
        self.toTensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.is_test = is_test
        self._init_ids()

    def _init_ids(self):
        data = self.args.test_input_dir if self.is_test else self.args.input_dir
        print(f"Loading data from {data}")
        file_names = os.listdir(data)
        self.data_ids += [os.path.join(data, id) for id in file_names]
        random.shuffle(self.data_ids)
        print(f"Total number of {'test' if self.is_test else 'training'} data: {len(self.data_ids)}")

    def _crop_patch(self, img_1, img_2):
        H, W = img_1.shape[0], img_1.shape[1]

        if H < self.args.patch_size or W < self.args.patch_size:
            img_1 = cv2.resize(img_1, (self.args.patch_size, self.args.patch_size), interpolation=cv2.INTER_LINEAR)
            img_2 = cv2.resize(img_2, (self.args.patch_size, self.args.patch_size), interpolation=cv2.INTER_LINEAR)
            H, W = img_1.shape[0], img_1.shape[1]

        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_data_gt(self, data_name):
        return data_name.replace('input', 'gt') if self.is_test else data_name.replace('input', 'gt')

    def __getitem__(self, index):
        sample = self.data_ids[index]
        degrad_img = crop_img(np.array(Image.open(sample).convert('RGB')), base=16)
        clean_name = self._get_data_gt(sample)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

        if not self.is_test:
            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
            degrad_patch = self.toTensor(degrad_patch)
            clean_patch = self.toTensor(clean_patch)
        else:
            degrad_patch = self.toTensor(degrad_img)
            clean_patch = self.toTensor(clean_img)

        return degrad_patch, clean_patch, sample

    def __len__(self):
        return len(self.data_ids)

# ===============================
# Function to Adjust Sample Selection Weight
# ===============================
def adjust_sample_selection_weight(epoch, total_epochs):
    # Dynamically adjust complexity and uncertainty weights
    if epoch < total_epochs * 0.5:
        alpha, beta = 0.7, 0.2  # Focus more on simple samples initially
    elif epoch < total_epochs * 0.8:
        alpha, beta = 0.5, 0.4  
    else:
        alpha, beta = 0.3, 0.6  
    return alpha, beta

# ===============================
# Calculate Image Complexity and Uncertainty using ViT and CLIP
# ===============================
def get_image_complexity_and_uncertainty(image_path, device, vit_model, clip_model, feature_extractor, clip_processor):
    # Move models to appropriate device
    vit_model.to(device)
    clip_model.to(device)

    # Open image and prepare for models
    image = Image.open(image_path).convert('RGB')

    # ViT feature extraction
    vit_inputs = feature_extractor(images=image, return_tensors="pt")
    vit_inputs = vit_inputs['pixel_values'].to(device)
    vit_outputs = vit_model(pixel_values=vit_inputs)
    vit_features = vit_outputs.last_hidden_state.mean(dim=1)
    complexity_score_vit = torch.var(vit_features).item()

    # CLIP feature extraction
    clip_inputs = clip_processor(images=image, return_tensors="pt")
    clip_inputs = clip_inputs['pixel_values'].to(device)
    clip_outputs = clip_model.get_image_features(clip_inputs)
    complexity_score_clip = torch.var(clip_outputs).item()

    # Combine both scores for a comprehensive complexity score
    combined_complexity_score = 0.5 * complexity_score_vit + 0.5 * complexity_score_clip

    # Calculate uncertainty score using CLIP output
    uncertainty_score = torch.var(clip_outputs).item()

    return combined_complexity_score, uncertainty_score

# ===============================
# Dynamic Dataset Distillation with Small Batch Updates and Sample Complexity
# ===============================
def distill_dataset_with_vit_clip_partial(args, teacher_model, student_model, epoch, total_epochs, vit_model, clip_model, feature_extractor, clip_processor, batch_size=16):
    dataset = ImageDataset(args)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    distilled_images, distilled_gt_images = [], []

    # Adjust weights for sample selection
    alpha, beta = adjust_sample_selection_weight(epoch, total_epochs)
    print(f"Epoch {epoch + 1}: alpha={alpha}, beta={beta}")

    # Only distill a portion of the data each time
    for i, (degrad_patch, clean_patch, sample) in enumerate(data_loader):
        if i >= batch_size:
            break  # Only take a limited number of batches for distillation to keep updates small

        # Calculate complexity and uncertainty scores
        complexity, uncertainty = get_image_complexity_and_uncertainty(sample[0], torch.device('cuda'), vit_model, clip_model, feature_extractor, clip_processor)
        score = alpha * complexity + beta * uncertainty

        if score < 0.5:  
            with torch.no_grad():
                distilled_images.append(degrad_patch.cpu())
                distilled_gt_images.append(clean_patch.cpu())

    if distilled_images:
        distilled_images = torch.cat(distilled_images, dim=0)
        distilled_gt_images = torch.cat(distilled_gt_images, dim=0)
    else:
        print("No samples selected for distillation in this batch.")
        distilled_images = torch.Tensor()
        distilled_gt_images = torch.Tensor()

    print(f"Total distilled images for this update: {len(distilled_images)}")
    return distilled_images, distilled_gt_images

# ===============================
# Evaluate Model Performance
# ===============================
def evaluate_model_performance(model, test_loader):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_samples = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for degrad_patch, clean_patch, _ in test_loader:
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)

            output = model(degrad_patch)

            output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            clean_np = clean_patch.squeeze().cpu().numpy().transpose(1, 2, 0)

            psnr_value = compare_psnr(clean_np, output_np, data_range=1.0)
            ssim_value = compare_ssim(clean_np, output_np, multichannel=False, channel_axis=-1, data_range=1.0)

            total_psnr += psnr_value
            total_ssim += ssim_value
            num_samples += 1

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print(f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

# ===============================
# Train Student Model with Frequent Small Updates
# ===============================
def train_student_model(args, student_model, distilled_images, distilled_gt_images, teacher_model, test_loader, optimizer, scheduler, scaler):
    student_model.train()
    teacher_model.load_state_dict(torch.load('/data/SDA/suxin/kd-data/motion_deblurring.pth'), strict=False)
    teacher_model.eval()
    device = next(student_model.parameters()).device

    total_loss = 0
    for i in range(len(distilled_images)):
        student_input = distilled_images[i].unsqueeze(0).to(device)
        gt_image = distilled_gt_images[i].unsqueeze(0).to(device)

        with torch.no_grad():
            teacher_output = teacher_model(student_input)

        with torch.cuda.amp.autocast():
            student_output = student_model(student_input)
            distillation_loss = F.mse_loss(student_output, teacher_output)
            supervision_loss = F.l1_loss(student_output, gt_image)
            
            alpha = min(1.0, 0.1 + args.current_epoch / args.total_epochs)
            total_loss_batch = alpha * distillation_loss + (1 - alpha) * supervision_loss

        
        scaler.scale(total_loss_batch).backward()

        if (i + 1) % 4 == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += total_loss_batch.item()

    avg_loss = total_loss / len(distilled_images)
    print(f"Training Loss for epoch {args.current_epoch + 1}: {avg_loss:.4f}")

    scheduler.step()

    # Evaluate model performance every epoch
    avg_psnr, avg_ssim = evaluate_model_performance(student_model, test_loader)

    return avg_psnr

# ===============================
# Main Function
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Restormer Image Deraining')
    parser.add_argument('--input_dir', type=str, default='/data/SDA/suxin/Multitask/GoPro/train/input', help='Path to input images directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for training')
    parser.add_argument('--syn_data_path', type=str, default='/data/SDA/suxin/kd-data/model_gopro/', help='Path to save distilled synthetic data')
    parser.add_argument('--total_epochs', type=int, default=300, help='Total number of training epochs')
    parser.add_argument('--test_input_dir', type=str, default='/data/SDA/suxin/Multitask/GoPro/test/input', help='Path to test input images directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = Restormer().to(device)
    student_model = Restormer().to(device)
    student_model.load_state_dict(torch.load('/data/SDA/suxin/kd-data/model_gopro/best_student_model.pth'))
    print("Start training student model.")
    
    test_dataset = ImageDataset(args, is_test=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load ViT and CLIP models
    vit_model_path = "/data/SDA/suxin/kd-data/vit-base-patch16-224"
    clip_model_path = "/data/SDA/suxin/kd-data/clip-vit-base-patch32"
    feature_extractor = ViTImageProcessor.from_pretrained(vit_model_path)
    vit_model = ViTModel.from_pretrained(vit_model_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path)

    best_psnr = 0
    best_model_path = None

    optimizer = AdamW(student_model.parameters(), lr=3e-4, weight_decay=1e-2)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.total_epochs - 5, eta_min=1e-6)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.total_epochs):
        print(f"Starting epoch {epoch + 1}")
        args.current_epoch = epoch

        distilled_images, distilled_gt_images = distill_dataset_with_vit_clip_partial(
            args, teacher_model, student_model, epoch, args.total_epochs, vit_model, clip_model, feature_extractor, clip_processor)

        if distilled_images.size(0) > 0:
            avg_psnr = train_student_model(args, student_model, distilled_images, distilled_gt_images, teacher_model, test_loader, optimizer, scheduler, scaler)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_path = os.path.join(args.syn_data_path, 'best_student_model.pth')
                torch.save(student_model.state_dict(), save_path)
                print(f"New best model saved with PSNR: {best_psnr:.4f}")

        print(f"Epoch {epoch + 1} completed.")
