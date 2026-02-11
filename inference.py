from dinov2.models.vision_transformer import vit_small

model = vit_small()

for name, param in model.named_parameters():
    print("name, param, param.shape: ", name, param, param.shape)