import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np

dataset = dset.ImageFolder("data/faces", transform=transforms.Compose([
                            #transforms.RandomRotation(degrees=180),
                            transforms.Resize(size=64),
                            transforms.CenterCrop(size=64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]))
'''
counter = 834
for i in dataset:
    im = i[0]
    im.save("augmented/" + str(counter) + ".jpg")
    counter += 1
'''

manualSeed = 999
torch.manual_seed(manualSeed)

batch_size = 64
image_size = 256
nc = 3
nz = 100
ngf = 64
ndf = 64
n_epochs = 5
lr = 1e-6
beta1 = 0.5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
         nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
         nn.BatchNorm2d(ngf * 8),
         nn.ReLU(),
         nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ngf * 4),
         nn.ReLU(),
         nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ngf * 2),
         nn.ReLU(),
         nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ngf),
         nn.ReLU(),
         nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
         nn.Tanh()
      )
    
    self.optim = optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))

  def forward(self, input):
    return self.main(input)
  
gNet = Generator().to(device)
gNet.apply(weights_init)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
         nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
         nn.LeakyReLU(0.2, inplace=True),
         nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ndf * 2),
         nn.LeakyReLU(0.2, inplace=True),
         nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ndf * 4),
         nn.LeakyReLU(0.2, inplace=True),
         nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ndf * 8),
         nn.LeakyReLU(0.2, inplace=True),
         nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
         nn.Sigmoid()
      )
    
    self.optim = optim.Adam(self.parameters(), lr=lr, betas=(beta1, 0.999))
    
  def forward(self, input):
     return self.main(input)
  
dNet = Discriminator().to(device)
dNet.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

img_list = []
g_loss = []
d_loss = []
d_real_acc = []
d_fake_acc = []
g_acc = []
iters = 0

for epoch in range(n_epochs):
   for i, data in enumerate(dataloader, 0):
      
      # Update Discriminator
      dNet.zero_grad()
      real_cpu = data[0].to(device)
      b_size = real_cpu.size(0)
      label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

      output = dNet(real_cpu).view(-1)

      errD_real = criterion(output, label)

      errD_real.backward()
      D_x = output.mean().item()

      noise = torch.randn(b_size, nz, 1, 1, device=device)

      fake = gNet(noise)
      label.fill_(fake_label)

      output = dNet(fake.detach()).view(-1)

      errD_fake = criterion(output, label)

      errD_fake.backward()
      D_G_z1 = output.mean().item()

      errD = errD_real + errD_fake
      dNet.optim.step()

      # Update Generator
      gNet.zero_grad()
      label.fill_(real_label)
      output = dNet(fake).view(-1)
      errG = criterion(output, label)
      errG.backward()
      D_G_z2 = output.mean().item()
      gNet.optim.step()

      if i % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, n_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
      # Save losses for graph
      g_loss.append(errG.item())
      d_loss.append(errD.item())

      # Save discriminator accuracy for graph
      d_real_acc.append(D_x)
      d_fake_acc.append(D_G_z2)

      # Save generator accuracy for graph
      g_acc.append(1-D_G_z2)

      if (iters % 50 == 0) or ((epoch == n_epochs-1) and (i == len(dataloader)-1)):
        with torch.no_grad():
          fake = gNet(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

      iters += 1

real_batch = next(iter(dataloader))

# This graph illustrates the generator and discriminator's loss over time during the training process.
# The discriminator's loss is calculated as the sum of all binary cross-entropy loss for both the
#   real and fake batches: log(D(x)) + log(1-D(G(z)))
# The generator's loss is calculated as the binary cross-entropy loss of discriminator's result of the
#   generator's output (i.e. the generator is trying to fool the discriminator): log(D(G(z)))

# Plot loss over time
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_loss, label="G")
plt.plot(d_loss, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# The discriminator's accuracy is a measurement of how well it predicts if an image belongs to the
# real dataset or is one of the generator's fake images. Here we plot the accuracy over time for
# both the real and fake datasets. The generator's accuracy is the compliment of the discriminator's
# accuracy for the fake dataset.

# Plot discriminator accuracy over time
plt.figure(figsize=(10,5))
plt.title("Model Accuracy During Training")
plt.plot(d_real_acc, label="Discriminator Real")
plt.plot(d_fake_acc, label="Discriminator Fake")
plt.plot(g_acc, label="Generator")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Here we can visualize the improvements of the DCGAN network output over time. As the generator
# continues to learn how to trick the discriminator every iteration, the look of the images will
# improve as they become harder and harder for the discriminator to distinguish from real images.

# Plot changing fake images over time
fig = plt.figure(figsize=(1,1))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=34, repeat_delay=1000, repeat=True, blit=True)
HTML(ani.to_jshtml())

# Finally, a comparison of our finalized output images to the real images we provided to the model
# as training data.

# Plot real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1, 2, 0)))

# Plot fake images
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))

plt.show()