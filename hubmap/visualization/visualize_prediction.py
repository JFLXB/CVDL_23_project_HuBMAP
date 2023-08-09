from typing import Optional
from pathlib import Path
from enum import StrEnum
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import scienceplots as _
import itertools

from hubmap.experiments.load_data import make_expert_loader
from hubmap.experiments.load_data import make_annotated_loader
from hubmap.dataset import transforms as T
from hubmap.metrics import IoU

from checkpoints import CHECKPOINT_DIR


class ImageType(StrEnum):
    expert = "expert"
    annotated = "annotated"


def visualize_random_image(model, checkpoint_name: str, image_type: ImageType = ImageType.expert, seed: Optional[int] = None):
    plt.style.use(["science"])
    checkpoint = torch.load(Path(CHECKPOINT_DIR / checkpoint_name))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    image, target = get_random_image_and_target(image_type, seed)
    image = image.to(device)

    prediction = get_prediction(model, image).detach().cpu()
    image = image.cpu()

    iou = IoU(0)
    iou_score = iou(prediction, target)
    
    image = image[0].squeeze().permute(1, 2, 0)
    target = target[0].squeeze().permute(1, 2, 0)
    prediction = prediction[0].squeeze().permute(1, 2, 0)
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 5))
    ax[0].imshow(image)
    ax[0].imshow(target, alpha=0.4)
    ax[0].set_title(f"Ground Truth")
    
    ax[1].imshow(image)
    ax[1].imshow(prediction, alpha=0.4)
    ax[1].set_title(f"Prediction / IoU: {(iou_score.item() * 100):.2f}%")
    # fig.suptitle(f"IoU: {(iou_score.item() * 100):.2f}%")
    plt.tight_layout()
    return fig


def get_prediction(model, image):
    model.eval()
    with torch.no_grad():
        print(image.size())
        prediction = model(image)
        probs = F.softmax(prediction, dim=1)
        classes = torch.argmax(probs, dim=1, keepdims=True)
        classes_per_channel = torch.zeros_like(prediction)
        classes_per_channel.scatter_(1, classes, 1)
    return classes_per_channel


def get_random_image_and_target(image_type: ImageType, seed: Optional[int] = None):
    # TODO: fix and make generic
    IMG_DIM = 64
    BATCH_SIZE = 100
    train_transformations = T.Compose(
        [T.ToTensor(), T.Resize((IMG_DIM, IMG_DIM)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transformations = T.Compose(
        [T.ToTensor(), T.Resize((IMG_DIM, IMG_DIM)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    loader = None
    if image_type == ImageType.expert:
        loader = make_expert_loader(train_transformations, test_transformations, train_ratio=1.0)
    elif image_type == ImageType.annotated:
        loader = make_annotated_loader(train_transformations, test_transformations, train_ratio=1.0)
    else:
        raise ValueError(f"Image type {image_type} is not supported.")
    
    train_loader, _ = loader(BATCH_SIZE)
    idx = random.randint(0, BATCH_SIZE - 1)
    for i, (image, target) in enumerate(train_loader):
        if i == idx:
            break
    print(image.size())
    return image, target
    
    