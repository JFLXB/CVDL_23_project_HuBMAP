import torch
import torch.functional as F
import matplotlib.pyplot as plt



def visualize_results(f: int, t: int, data, model, device):
    samples = []
    for i in range(f, t):
        wsi, target = data[i]
        
        model.eval()
        with torch.no_grad():
            pred = model(wsi.unsqueeze(0).to(device))
            logits = pred.detach().cpu()
            mask = F.sigmoid(logits)
        
        samples.append((
                wsi.squeeze(0).permute(1, 2, 0),
                target.permute(1, 2, 0),
                mask.squeeze(0).permute(1, 2, 0)
        ))
        
    fig_height = 2.5 * len(samples)
    fig, axs = plt.subplots(nrows=len(samples), ncols=2, figsize=(5,  fig_height))
    for i, (wsi, target, mask) in enumerate(samples):
        if i == 0:
            axs[i][0].set_title("Ground Truth")
            axs[i][1].set_title("Prediction")

        axs[i][0].imshow(wsi)
        axs[i][0].imshow(target, alpha=0.3)

        axs[i][1].imshow(wsi)
        axs[i][1].imshow(mask, alpha=0.5)

    plt.tight_layout()
#     plt.suptitle("Overlay of Ground Truth and Prediction (Sample from Training Data)")
    plt.show()