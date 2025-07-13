from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import os
import argparse
import copy
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import utils
import models
import urllib
from torch import nn
from torchvision import transforms as pth_transforms, datasets
from sklearn.manifold import TSNE
import sys
import numpy as np


parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=1` all the time for k-NN evaluation.""")
parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
                    help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base-size models with [CLS] token when doing linear classification.""")
parser.add_argument('--batch_size_per_gpu', default=128,
                    type=int, help='Per-GPU batch-size')
parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                    help='Number of NN to use. 20 is usually working the best.')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='Temperature used in the voting coefficient')
parser.add_argument('--pretrained_weights', default='/fp/projects01/ec35/homes/ec-thallesss/representation_learning/src/methods/SOVE/runs/Dec28_22-03-28_nid005056/checkpoint0399.pth', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                    help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
                                                                      'vit_large', 'swin_tiny', 'swin_small', 'swin_base', 'swin_large', 'resnet50', 'resnet101', 'dalle_encoder'], help='Architecture.')
parser.add_argument('--patch_size', default=16, type=int,
                    help='Patch resolution of the model.')
parser.add_argument('--window_size', default=7, type=int,
                    help='Window size of the model.')
parser.add_argument("--checkpoint_key", default="teacher", type=str,
                    help='Key to use in the checkpoint (example: "teacher")')
parser.add_argument('--num_workers', default=10, type=int,
                    help='Number of data loading workers per GPU.')
parser.add_argument('--data_path', default='/fp/projects01/ec35/data', type=str,
                    help='Please specify path to the ImageNet data.')

args = parser.parse_args()

# ============ preparing data ... ============

transform = pth_transforms.Compose([
    pth_transforms.Resize(256, interpolation=3),
    pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# dataset = datasets.CIFAR10(
#     root=args.data_path, train=True, transform=transform)
dataset = datasets.CIFAR100(
    root=args.data_path, train=False, transform=transform)

targets = np.asarray(dataset.targets)
print("Targets shape:", len(targets))

loader = torch.utils.data.DataLoader(
    dataset,
    sampler=None,
    batch_size=256,
    shuffle=False,
    num_workers=7,
    pin_memory=True,
    drop_last=False,
)


print(
    f"Data loaded with {len(dataset)}.")

# ============ building network ... ============
model = models.__dict__[args.arch](
    patch_size=args.patch_size,
    num_classes=0,
    use_mean_pooling=False)

print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
model.cuda()
utils.load_pretrained_weights(
    model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
model.eval()

features = torch.zeros(size=(len(dataset), 768)).cuda()
index = 0
for batch_images, batch_labels in loader:
    batch_images = batch_images.cuda()
    # batch_labels = batch_labels.cuda()
    with torch.no_grad():
        embeds = model(batch_images)
        B, D = embeds.shape
        features[index: index + B, :] = embeds
        index += B
    
features = features.cpu().numpy()
X_embedded = TSNE(n_components=3, learning_rate='auto',
                  init='random', perplexity=30.0).fit_transform(features)

# fig, axs = plt.subplots(figsize=(8, 8), nrows=1)
# cdict = {0: "gray", 1: 'red', 2: 'blue', 3: 'green', 4: 'gold', 5: 'magenta', 6: 'indigo', 7: "brown", 8: "orange", 9: "tan"}

# axs.scatter(X_embedded[:, 0], X_embedded[:, 1], c=targets, cmap='tab20', s=4, alpha=0.9)

# axs.set_xticks([])
# axs.set_yticks([])
    
# # Iterating over all the axes in the figure
# # and make the Spines Visibility as False
# for pos in ['right', 'top', 'bottom', 'left']:
#     plt.gca().spines[pos].set_visible(False)
    
# Create a sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

# Convert spherical coordinates to Cartesian coordinates
xx = X_embedded[:, 0]
yy = X_embedded[:, 1]
zz = X_embedded[:, 2]

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
ax.scatter(xx, yy, zz, c=targets, marker='o', alpha=0.6)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Points on a 3D Sphere')

# Set equal aspect ratio for all axes
ax.set_box_aspect([1, 1, 1])
plt.savefig("sop_embeddings.pdf", dpi=150)
