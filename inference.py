import torch
import requests
from PIL import Image 
from dinov2.models.vision_transformer import vit_small
from torchvision import transforms 

# loading the model
model = vit_small(patch_size=14)
# integrating the model with weights 
checkpoint = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
state_dict = torch.hub.load_state_dict_from_url(checkpoint)
for name, param in model.named_parameters():
    print("name, param.shape: ", name, param.shape)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

transformations = transforms.Compose(
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
)

pixel_values = transformations(image).unsqueeze(0) # inserting batch dimension
outputs = model.forward_features(pixel_values)
print("Outputs: ", outputs)