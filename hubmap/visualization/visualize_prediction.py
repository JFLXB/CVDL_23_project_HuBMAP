from typing import Optional
from pathlib import Path
from enum import StrEnum
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scienceplots as _

from skimage.color import label2rgb
from hubmap.experiments.load_data import make_expert_loader
from hubmap.experiments.load_data import make_annotated_loader
from hubmap.dataset import transforms as T
from hubmap.metrics import IoU

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
        # probs = F.sigmoid(prediction)
        print(prediction.size())
        pred_mask = torch.max(prediction, dim=1)[0].squeeze()
        print(pred_mask.unique())
        classes = torch.argmax(prediction, dim=1, keepdims=True)
        classes_per_channel = torch.zeros_like(prediction)
        classes_per_channel.scatter_(1, classes, 1)\
            
    plt.imshow(pred_mask)

    image = image.cpu()

    iou = IoU(1)
    iou_score = iou(classes_per_channel, target.unsqueeze(0))

    colors = {"blood_vessel": "red", "glomerulus": "blue", "unsure": "yellow"}
    titles = {
        "blood_vessel": "Blood Vessel",
        "glomerulus": "Glomerulus",
        "unsure": "Unsure",
    }

    image_np = image.permute(1, 2, 0).squeeze().numpy()
    target_argmax = target.argmax(dim=0)
    target_np = target_argmax.numpy()
    
    if not overlay:
        target_mask_img = label2rgb(
            target_np, image=None, bg_label=0, colors=colors.values(), kind="overlay"
        )
        pred_mask_np = pred_mask.numpy()
        pred_mask_img = label2rgb(
            pred_mask_np, image=None, bg_label=0, colors=colors.values(), kind="overlay"
        )

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6, 2.5))
        axs[0].imshow(image_np)
        axs[0].set_title(f"Image")
        axs[1].imshow(target_mask_img)
        axs[1].set_title(f"Ground Truth")
        axs[2].imshow(pred_mask_img)
        axs[2].set_title(f"Prediction")
    else:
        assert False, "Not implemented yet."
        saturation = 0.0 if grayscale else 1.0
        target_mask_img = label2rgb(
            target_np,
            image=image_np,
            bg_label=3,
            colors=colors.values(),
            kind="overlay",
            saturation=saturation,
            alpha=0.4,
        )
        pred_mask_np = pred_mask.numpy()
        pred_mask_img = label2rgb(
            pred_mask_np,
            image=image_np,
            bg_label=3,
            colors=colors.values(),
            kind="overlay",
            saturation=saturation,
            alpha=0.4,
        )

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))
        axs[0].imshow(target_mask_img)
        axs[0].set_title(f"Ground Truth")
        axs[1].imshow(pred_mask_img)
        axs[1].set_title(f"Prediction")

    if legend:
        blood_vessel_patch = mpatches.Patch(
            facecolor=colors["blood_vessel"],
            label=titles["blood_vessel"],
            edgecolor="black",
        )
        glomerulus_patch = mpatches.Patch(
            facecolor=colors["glomerulus"],
            label=titles["glomerulus"],
            edgecolor="black",
        )
        unsure_patch = mpatches.Patch(
            facecolor=colors["unsure"], label=titles["unsure"], edgecolor="black"
        )
        handles = [blood_vessel_patch, glomerulus_patch, unsure_patch]
        # fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3
        )

    if title:
        fig.suptitle(f"{checkpoint_name} / IoU: {(iou_score.item() * 100):.2f}%")
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
