import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt 

from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

img_path = "/home/ryqc/Downloads/pug.jpg"
image = Image.open(img_path).convert("RGB")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transforms_img = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(imagenet_mean, imagenet_std),
])


model = efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
#print(model.features)

model.classifier[1] = torch.nn.Linear(in_features=in_features, out_features=10)

state_dict = torch.load("/home/ryqc/projects/python_projects/Image-Video-Gen/cnn/efficient_net_tests/experiments_cnnAblation/efficient_net_epoch_39.pt", map_location="cpu")
model.load_state_dict(state_dict["model"])
model.to(device)
model.eval()
# add 1 dim to image to feeed it to the efficientnet_b0
x = transforms_img(image).unsqueeze(0).to(device)


activations = {}
def activation_hook(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook


hooks = []
for i in range(6):
    hooks.append(
        model.features[i].register_forward_hook(activation_hook(f"block{i}"))
    )

# Inference 
with torch.no_grad():
    output = model(x)
    probs = torch.softmax(output, dim=1)
    pred = probs.argmax(dim=1).item()

#for q,s in activations.items():
    #print(f"keys:{q} ----- values:{s}")


act = activations["block2"][0]  # [C, H, W]

num_maps = min(8, act.shape[0])
fig, axes = plt.subplots(1, num_maps, figsize=(15, 3))

for i in range(num_maps):
    axes[i].imshow(act[i], cmap="viridis")
    axes[i].axis("off")

plt.show()

for h in hooks:
    h.remove()
