from torch import optim
from .SegNet import *
from .dataload import *
import torch

num_epochs = 1000
lr = 0.0001
val_check = 5

# We have 3 input channels (rgb) and 6 classes we want to semantically segment
model = UNet(n_channels=3, n_classes=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Replace with the path to your scene file
SCENE_FILEPATH = '/media/phong/data/dataset/replica/apartment_0/mesh.ply'
BATCH_SIZE = 4
extractor = ImageExtractor(SCENE_FILEPATH, output=['rgba', 'semantic'])

dataset = SemanticSegmentationDataset(extractor,
    transforms=T.Compose([T.ToTensor()]))

# Create a Dataloader to batch and shuffle our data
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in dataloader:
        imgs = batch['rgb']
        true_masks = batch['truth']

        # Move the images and truth masks to the proper device (cpu or gpu)
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)

        # Get the model prediction
        masks_pred = model(imgs)

        # Evaluate the loss, which is Cross-Entropy in our case
        loss = criterion(masks_pred, true_masks)
        epoch_loss += loss.item()

        # Update the model parameters
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

    # Evaluate the model on validation set
    if epoch % val_check == 0:
        print(f"iter: {epoch}, train loss: {epoch_loss}")

def show_batch(sample_batch):
    def show_row(imgs, batch_size, img_type):
        plt.figure(figsize=(12, 8))
        for i, img in enumerate(imgs):
            ax = plt.subplot(1, batch_size, i + 1)
            ax.axis("off")
            if img_type == 'rgb':
                plt.imshow(img.numpy().transpose(1, 2, 0))
            elif img_type in ['truth', 'prediction']:
                plt.imshow(img.numpy())

        plt.show()

    batch_size = len(sample_batch['rgb'])
    for k in sample_batch.keys():
        show_row(sample_batch[k], batch_size, k)

# Testing
with torch.no_grad():
    model.to('cpu')
    model.eval()
    _, batch = next(enumerate(dataloader))
    mask_pred = model(batch['rgb'])
    mask_pred = F.softmax(mask_pred, dim=1)
    mask_pred = torch.argmax(mask_pred, dim=1)

    batch['prediction'] = mask_pred

    show_batch(batch)