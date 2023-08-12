from typing import Optional
from pathlib import Path
from enum import StrEnum
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots as _
import numpy as np
import PIL

from skimage.color import label2rgb
from hubmap.experiments.load_data import make_expert_loader
from hubmap.experiments.load_data import make_annotated_loader
from hubmap.dataset import transforms as T
from hubmap.dataset import label2id, label2title
from hubmap.metrics import IoU

from hubmap.visualization.visualize_mask import mask_to_rgb, mask_to_rgba

from checkpoints import CHECKPOINT_DIR


class ImageType(StrEnum):
    expert = "expert"
    annotated = "annotated"


def visualize_image(
    model,
    checkpoint_name: str,
    image: torch.Tensor,
    target: torch.Tensor,
    transforms,
    pred_idx: int,
    overlay: bool = False,
    grayscale: bool = False,
    legend: bool = True,
    title: bool = True,
    activation_fun=None,
    cmap: dict = None,
    saturation: float = 0.7,
    alpha: float = 0.6,
):
    plt.style.use(["science"])
    checkpoint = torch.load(Path(CHECKPOINT_DIR / checkpoint_name))

    if (len(image.size()) != 3) or (len(target.size()) != 3):
        raise ValueError(
            f"Image and target must have 3 dimensions, but got {len(image.size())} and {len(target.size())}."
        )

    device = "cpu"
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    image = image.to(device)

    if transforms:
        img, target = transforms(image, target)

    model.eval()
    with torch.no_grad():
        prediction = model(img.unsqueeze(0))
        prediction = prediction[pred_idx] if pred_idx is not None else prediction

        if activation_fun:
            prediction = activation_fun(prediction)

        classes = torch.argmax(prediction, dim=1, keepdims=True)
        classes_per_channel = torch.zeros_like(prediction)
        classes_per_channel.scatter_(1, classes, 1)
        classes_per_channel = classes_per_channel.squeeze(0)

    image = image.cpu()

    iou = IoU()
    iou_score = iou(classes_per_channel, target).item()

    iou_blood_vessel = IoU(class_index=label2id["blood_vessel"])
    iou_blood_vessel_score = iou_blood_vessel(classes_per_channel, target).item()
    iou_glomerulus = IoU(class_index=label2id["glomerulus"])
    iou_glomerulus_score = iou_glomerulus(classes_per_channel, target).item()
    iou_unsure = IoU(class_index=label2id["unsure"])
    iou_unsure_score = iou_unsure(classes_per_channel, target).item()
    iou_background = IoU(class_index=label2id["background"])
    iou_background_score = iou_background(classes_per_channel, target).item()

    # iou_blood_vessel_score = 0
    # iou_glomerulus_score = 0
    # iou_unsure_score = 0
    # iou_background_score = 0

    colors = {
        "blood_vessel": "tomato",
        "glomerulus": "dodgerblue",
        "unsure": "palegreen",
        "background": "black",
    }
    colors = colors if cmap is None else cmap
    cmap = {label2id[l]: colors[l] for l in colors.keys()}

    image_np = image.permute(1, 2, 0).squeeze().numpy()

    if not overlay:
        target_mask_rgb = mask_to_rgb(target, color_map=cmap, bg_channel=-1)
        pred_mask_rgb = mask_to_rgb(classes_per_channel, color_map=cmap, bg_channel=-1)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6, 2.5))
        axs[0].imshow(image_np)
        axs[0].set_title(f"Image")
        axs[1].imshow(target_mask_rgb.permute(1, 2, 0))
        axs[1].set_title(f"Ground Truth")
        axs[2].imshow(pred_mask_rgb.permute(1, 2, 0))
        axs[2].set_title(f"Prediction")
    else:
        raise NotImplementedError
        # image = PIL.Image.fromarray(np.uint8(image_np) * 255)
        # image = image.convert("LA") if grayscale else image.convert("RGBA")
        # image.putalpha(int(255 * saturation))

        # target_mask_rgb = mask_to_rgba(
        #     target, color_map=cmap, bg_channel=3, alpha=1.0
        # )
        # pred_mask_rgb = mask_to_rgba(
        #     classes_per_channel, color_map=cmap, bg_channel=3, alpha=alpha
        # )

        # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))
        # axs[0].imshow(image, cmap="gray") if grayscale else axs[0].imshow(image_np)
        # axs[0].imshow(target_mask_rgb.permute(1, 2, 0))
        # axs[0].set_title(f"Ground Truth")
        # axs[1].imshow(image, cmap="gray") if grayscale else axs[0].imshow(image_np)
        # axs[1].imshow(pred_mask_rgb.permute(1, 2, 0), alpha=0.5)
        # axs[1].set_title(f"Prediction")

    if legend:
        blood_vessel_patch = mpatches.Patch(
            facecolor=colors["blood_vessel"],
            label=f"{label2title['blood_vessel']}\n(IoU: {iou_blood_vessel_score * 100:.2f})",
            edgecolor="black",
        )
        glomerulus_patch = mpatches.Patch(
            facecolor=colors["glomerulus"],
            label=f"{label2title['glomerulus']}\n(IoU: {iou_glomerulus_score * 100:.2f})",
            edgecolor="black",
        )
        unsure_patch = mpatches.Patch(
            facecolor=colors["unsure"],
            label=f"{label2title['unsure']}\n(IoU: {iou_unsure_score * 100:.2f})",
            edgecolor="black",
        )
        background_patch = mpatches.Patch(
            facecolor=colors["background"],
            label=f"{label2title['background']}\n(IoU: {iou_background_score * 100:.2f})",
            edgecolor="black",
        )
        handles = [blood_vessel_patch, glomerulus_patch, unsure_patch, background_patch]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4
        )

    if title:
        fig.suptitle(f"{checkpoint_name} / IoU: {(iou_score * 100):.2f}")
    else:
        print("IoU: ", iou_score.item())
    fig.tight_layout()
    return fig


def visualize_random_image(
    model,
    checkpoint_name: str,
    image_type: ImageType = ImageType.expert,
    seed: Optional[int] = None,
    pred_idx: int = None,
):
    plt.style.use(["science"])
    checkpoint = torch.load(Path(CHECKPOINT_DIR / checkpoint_name))

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    image, target = get_random_image_and_target(image_type, seed)
    image = image.to(device)

    prediction = get_prediction(model, image, pred_idx).detach().cpu()
    print(prediction.size())
    image = image.cpu()

    iou = IoU(1)
    iou_score = iou(prediction, target)

    image = image[0].squeeze().permute(1, 2, 0)
    target = target[0].squeeze().permute(1, 2, 0)
    prediction = prediction[0].permute(1, 2, 0)

    fig, ax = plt.subplots(1, 3, figsize=(6, 5))
    ax[0].imshow(image)
    ax[0].set_title(f"Image")
    ax[1].imshow(target.argmax(dim=2, keepdims=True))
    ax[1].set_title(f"Ground Truth")
    ax[2].imshow(prediction.argmax(dim=2, keepdims=True))
    ax[2].set_title(f"Prediction / IoU: {(iou_score.item() * 100):.2f}%")
    # fig.suptitle(f"IoU: {(iou_score.item() * 100):.2f}%")
    plt.tight_layout()
    return fig


def get_prediction(model, image, pred_idx=None):
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        prediction = prediction[pred_idx] if pred_idx is not None else prediction
        probs = F.softmax(prediction, dim=1)
        classes = torch.argmax(probs, dim=1, keepdims=True)
        classes_per_channel = torch.zeros_like(prediction)
        classes_per_channel.scatter_(1, classes, 1)
    return classes_per_channel


def get_random_image_and_target(image_type: ImageType, seed: Optional[int] = None):
    # TODO: fix and make generic
    IMG_DIM = 128
    BATCH_SIZE = 100
    train_transformations = T.Compose(
        [
            T.ToTensor(),
            # T.AddBackgroundToMask(),
            T.Resize((IMG_DIM, IMG_DIM)),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_transformations = T.Compose(
        [
            T.ToTensor(),
            # T.AddBackgroundToMask(),
            T.Resize((IMG_DIM, IMG_DIM)),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    loader = None
    if image_type == ImageType.expert:
        loader = make_expert_loader(
            train_transformations, test_transformations, train_ratio=1.0
        )
    elif image_type == ImageType.annotated:
        loader = make_annotated_loader(
            train_transformations, test_transformations, train_ratio=1.0
        )
    else:
        raise ValueError(f"Image type {image_type} is not supported.")

    train_loader, _ = loader(BATCH_SIZE)
    idx = random.randint(0, BATCH_SIZE - 1)
    for i, (image, target) in enumerate(train_loader):
        if i == idx:
            break
    # print(image.size())
    return image, target
