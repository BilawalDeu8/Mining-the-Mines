import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchvision.transforms import Resize
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from pprint import pprint
from torch.utils.data import Dataset, DataLoader, random_split

class DataSource(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_filenames = [filename for filename in sorted(os.listdir(os.path.join(root_dir, '/path'))) if filename.endswith('.tif')]
        self.mask_filenames = [filename for filename in sorted(os.listdir(os.path.join(root_dir, '/path'))) if filename.endswith('.tif')]
        assert len(self.image_filenames) == len(self.mask_filenames), "Number of images and masks must match"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, '/path', self.image_filenames[idx])
        mask_path = os.path.join(self.root_dir, '/path', self.mask_filenames[idx])
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')  # Ensure mask is in grayscale

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return {'image': image, 'mask': mask}

# Define transformations for images and masks
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Converts PIL Image or ndarray to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

def to_binary_mask(x):
    return (x > 0).float()

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(to_binary_mask)  # Using a named function instead of lambda
])

# Create dataset instances with separate transformations for images and masks
dataset = DataSource(root_dir='output_images', transform=image_transform, target_transform=mask_transform)

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

# Setup data loaders
n_cpu = os.cpu_count() or 1  # Handle case where cpu_count is None
print(f"Number of CPUs available: {n_cpu}")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(len(train_dataset) + len(valid_dataset) + len(test_dataset))

import numpy as np
import matplotlib.pyplot as plt

def normalize_image(img):
    max_vals = np.max(img, axis=(0, 1), keepdims=True)
    min_vals = np.min(img, axis=(0, 1), keepdims=True)
    normalized_img = (img - min_vals) / (max_vals - min_vals)
    return normalized_img

def show_sample(sample):
    image_tensor = sample['image'].permute(1, 2, 0).numpy()  # Convert to NumPy array and permute to HWC
    normalized_image = normalize_image(image_tensor)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(normalized_image)  # Display normalized image
    plt.title('Normalized Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(sample['mask'].squeeze(), cmap='gray')  # Remove the channel dimension if necessary
    plt.title('Mask')
    plt.show()

# Visualize samples from the train, valid, and test sets
show_sample(train_dataset[531])
show_sample(valid_dataset[35])
show_sample(test_dataset[91])

class Mines(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres 
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # dice loss
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch):
        image = batch["image"]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
    
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
    
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }



    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset and then compute IoU score. 
        # There was high difference between dataset_iou and per_image_iou scores
        # because of high number of images without target class (black satellite images due to factors like cloud cover). Thus, we manually removed around 400 of these.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch)  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

model = Mines("FPN", "resnet34", in_channels=3, out_classes=1)

trainer = pl.Trainer(
    max_epochs=5,
)
trainer.log_every_n_steps = 1
trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)

#validation dataset
valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
pprint(valid_metrics)

# run test dataset
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)

