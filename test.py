from models_vit_nobias2 import vit_base_patch16 
import torch 

input = torch.randn(6, 3, 224, 224)


model = vit_base_patch16()
out = model(input)
print(out.size())


'''
for i in range(len(model.blocks)):
    print(model.blocks[i].attn.scale)
'''