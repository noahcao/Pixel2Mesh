import torch

from models.classifier import Classifier
from options import options


options.model.backbone = "vgg16"
model = Classifier(options.model, 1000)
state_dict = torch.load("checkpoints/debug/migration/400400_000080.pt")
model.load_state_dict(state_dict["model"])
torch.save(model.nn_encoder.state_dict(), "checkpoints/debug/migration/vgg16-p2m.pth")
