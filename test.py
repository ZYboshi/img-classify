import torch
from PIL import Image
from torchvision import transforms

from network import Net

def get_model(path):
    model=Net()
    model.load_state_dict(torch.load(path))
    return model

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28,28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.5,), (0.5,))
])


model=get_model("./checkpoints/model_40.pth")
img=Image.open("test.png").convert('RGB')
img=transform(img)
img=img.unsqueeze(0)
model.eval()
with torch.no_grad():
    ret=model(img)
_, predicted_class = torch.max(ret, 1)
print(f"识别到是数字： {predicted_class.item()}")

