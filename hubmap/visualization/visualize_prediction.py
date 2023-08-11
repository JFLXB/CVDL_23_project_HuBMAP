from typing import Optional
from pathlib import Path
from enum import StrEnum
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import scienceplots as _

from hubmap.experiments.load_data import make_expert_loader
from hubmap.experiments.load_data import make_annotated_loader
from hubmap.dataset import transforms as T
from hubmap.metrics import IoU

from checkpoints import CHECKPOINT_DIR


class ImageType(StrEnum):
    expert = "expert"
    annotated = "annotated"


def visualize_image(
    model, checkpoint_name: str, image: str, target: str, pred_idx: int
):
    plt.style.use(["science"])
    checkpoint = torch.load(Path(CHECKPOINT_DIR / checkpoint_name))

    image = torch.load(Path(image), map_location=torch.device("cpu"))
    target = torch.load(Path(target), map_location=torch.device("cpu"))

    device = "cpu"
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    image = image.to(device)
    prediction = get_prediction(model, image, pred_idx).detach().cpu()
    image = image.cpu()

    iou = IoU(0)
    iou_score = iou(prediction, target)

    image = image[0].squeeze().permute(1, 2, 0)
    target = target[0].permute(1, 2, 0)
    # image = image[0].permute(1, 2, 0)
    # target = target[0].permute(1, 2, 0)
    prediction = prediction[0].permute(1, 2, 0)

    fig, ax = plt.subplots(1, 3, figsize=(6, 5))
    ax[0].imshow(image)
    ax[0].set_title(f"Image")
    # ax[0].imshow(target, alpha=0.4)

    if target.size(2) < 3:
        diff = 3 - target.shape[2]
        ad = torch.zeros((target.size(0), target.size(1), diff))
        target = torch.cat((target, ad), 2)

    ax[1].imshow(target)
    ax[1].set_title(f"Ground Truth")

    # ax[1].imshow(image)
    # print(prediction.size(2))
    if prediction.size(2) < 3:
        diff = 3 - prediction.shape[2]
        ad = torch.zeros((prediction.size(0), prediction.size(1), diff))
        prediction = torch.cat((prediction, ad), 2)

    ax[2].imshow(prediction)
    ax[2].set_title(f"Prediction / IoU: {(iou_score.item() * 100):.2f}%")
    plt.tight_layout()
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
