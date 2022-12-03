#%%
from model import Generator
import torch
gen = Generator(embed_size=6)
gen.load_state_dict(torch.load("cgan0620.pth",map_location=torch.device('cpu')))
gen.eval()
# %% forward example
from matplotlib import pyplot as plt

def img_prc(img):
    img = img.detach().cpu()
    img = img.numpy()
    return img>0.5

torch.manual_seed(0)
z = torch.randn(10,100)
f = torch.ones(10,1)*0.2
input_gen = (z,f)

img_fake = gen(z,f)
img_fake = img_prc(img_fake)
plt.imshow(img_fake[0,0,0,::],cmap='gray')


# %% Convert to Torch Script
output = torch.jit.trace(gen,input_gen)

# %% check the reproduction quality
f = torch.ones(10,1)*0.3
img_fake = output(z,f)
img_fake = img_prc(img_fake)
plt.imshow(img_fake[0,0,0,::],cmap='gray')
# %% Serializing your script module to a file
output.save("gen-1.pt")