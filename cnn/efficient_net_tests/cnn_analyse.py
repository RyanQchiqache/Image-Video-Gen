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
for i in range(1, 9):
    hooks.append(
        model.features[i].register_forward_hook(activation_hook(f"block{i}"))
    )

# Inference 
with torch.no_grad():
    output = model(x)
    probs = torch.softmax(output, dim=1)
    pred = probs.argmax(dim=1).item()
    print(pred)

#for q,s in activations.items():
    #print(f"keys:{q} ----- values:{s}")


def vis_activations(activations, hooks_name, num_maps):
    act = activations[hooks_name][0]
    if num_maps > act.shape[0]:
        raise ValueError(f"choose number smaller or equal to: {act.shape[0]}")

    plt.figure(figsize=(15,5))
    
    for i in range(num_maps):
        plt.subplot(1, num_maps, i + 1)
        plt.imshow(act[i], cmap="viridis")
        plt.axis("off")
        plt.title(f"{hooks_name} | channel {i}")
    plt.tight_layout()
    plt.show()


for i in range(1, 9):
    vis_activations(activations, f"block{i}", 8)

for h in hooks:
    h.remove()


#----------------------
# Saliency_map
#---------------------

def saliency_map(X, y, model):

    model.eval()

    x_var = X.clone().detach().to(device)
   # x_var.requires_grad() = True
    y = y.to(device)

    scores = model(x_var)

    correct_scores = scores.gather(1, y.view(-1, 1)).squeeze()
    correct_scores.backward(torch.ones_like(correct_scores))
    saliency = x_var.grad.data.abs().max(dim=1)[0]

    return saliency




