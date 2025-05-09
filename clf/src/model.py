import timm
import torch
import torch.nn as nn

def build_model(
    model_name: str = "tf_efficientnetv2_s",
    num_classes: int = 92,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    multi_gpu: bool = True
):
    """
    Build a model for image classification.

    Args:
        model_name (str): Name of the model architecture.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use a pretrained model.
        freeze_backbone (bool): Whether to freeze the backbone layers.
        multi_gpu (bool): Whether to use multiple GPUs.

    Returns:
        model: torch.nn.Module: The constructed model.
        device: torch.device
    """
    # Load the model with the specified architecture and number of classes
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes
        )

    # Freeze the backbone layers if specified
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:  # timm models have a classifier layer
                param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Move the model to multiple GPUs if available and specified
    if multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model, device