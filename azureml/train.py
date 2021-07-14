"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import problem_unittests as tests
import torch
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from azureml.core.dataset import Dataset
from azureml.core.run import Run
# get the Azure ML run object
run = Run.get_context()
ws = run.experiment.workspace

print("Torch version:", torch.__version__)

#################################################
## Get Command-Line Arguments
#################################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='processed_celeba_small',
                    help='data directory')
parser.add_argument('--output_dir', type=str, default='./outputs',
                    help='output directory')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256,
                    help='number of words in a sequence')
parser.add_argument('--img_size', type=int, default=32,
                    help='image size')
parser.add_argument('--d_conv_dim', type=int, default=64,
                    help='discriminator convolution dim')
parser.add_argument('--g_conv_dim', type=int, default=64,
                    help='generator convolution dim')
parser.add_argument('--z_size', type=int, default=100,
                    help='z size')
parser.add_argument('--learning_rate', type=float, default=0.0002, 
                    help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, 
                    help='beta1 coefficient used for computing running averages of gradient and its square')
parser.add_argument('--beta2', type=float, default=0.999, 
                    help='beta2 coefficient used for computing running averages of gradient and its square')
args = parser.parse_args()

dataset_name = args.dataset_name
output_dir = args.output_dir
batch_size = args.batch_size
img_size = args.img_size
d_conv_dim = args.d_conv_dim
g_conv_dim = args.g_conv_dim
z_size = args.z_size
lr = args.learning_rate
beta1 = args.beta1
beta2 = args.beta2
n_epochs = args.num_epochs

print('Downloading Dataset ', dataset_name)
data_dir = './data/'
dataset = Dataset.get_by_name(ws, name=dataset_name)
os.makedirs(data_dir, exist_ok=True)
dataset.download(target_path=data_dir, overwrite=False)
print('Dataset downloaded successfully')

def get_dataloader(batch_size, image_size, data_dir='./data/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    
    # TODO: Implement function and return a dataloader
    data_loader = torch.utils.data.DataLoader(dataset = datasets.ImageFolder(
        data_dir, transform = 
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        ),
        batch_size = batch_size,
        shuffle = True)
    
    return data_loader

# Call your function and get a dataloader
celeba_train_loader = get_dataloader(batch_size, img_size)

def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    return x * (max - min) + min

# obtain one batch of training images
dataiter = iter(celeba_train_loader)
images, _ = dataiter.next() # _ for no labels

# check scaled range
# should be close to -1 to 1
img = images[0]
scaled_img = scale(img)

print('Min: ', scaled_img.min())
print('Max: ', scaled_img.max())

def conv(in_channels, out_channels, kernel_size, stride=2, padding = 1, negative_slope=0.2, batch_norm= True):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                       kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    layers.append(conv_layer)
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    layers.append(nn.LeakyReLU(negative_slope))
    
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
        
        self.conv1 = conv(3, conv_dim, 4, negative_slope=0.2, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, negative_slope=0.2)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, negative_slope=0.2)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, negative_slope=0.2)
        
        self.fc = nn.Linear(conv_dim*8*2*2, 1)
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # flatten
        x = x.view(-1, self.conv_dim*8*2*2)
        
        x = self.fc(x)
        return x
tests.test_discriminator(Discriminator)

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim

        self.fc = nn.Linear(z_size, conv_dim*8*2*2)
        
        self.t_conv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.t_conv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv3 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv4 = deconv(conv_dim, 3, 4, batch_norm=False)
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*8, 2, 2) # (batch_size, depth, 4, 4)
        
        x = self.t_conv1(x)
        x = F.relu(x)
        
        x = self.t_conv2(x)
        x = F.relu(x)
        
        x = self.t_conv3(x)
        x = F.relu(x)
        
        # last layer: tanh activation instead of relu
        x = self.t_conv4(x)
        x = torch.tanh(x)
        
        return x
tests.test_generator(Generator)

def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    # TODO: Apply initial weights to convolutional and linear layers
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.2)
        
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)

def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G
D, G = build_network(d_conv_dim, g_conv_dim, z_size)

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')

def real_loss(D_out):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)*0.9
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

# Create optimizers for the discriminator D and generator G
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

def train(D, G, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''
    
    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================
            
            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()
            if train_on_gpu:
                real_images = real_images.cuda()
                
            D_real = D(real_images)
            d_real_loss = real_loss(D_real)
            
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            
            fake_images = G(z)
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()
            
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            
            fake_images = G(z)
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake)
            
            g_loss.backward()
            g_optimizer.step()
            
            
            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))
                run.log('d_loss', np.float(d_loss.item()))
                run.log('g_loss', np.float(g_loss.item()))
        
        ## AFTER EACH EPOCH##    
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval() # for generating samples
        with torch.no_grad():
            samples_z = G(fixed_z)
            samples_z = samples_z.detach().cpu()
            samples.append(samples_z)
        G.train() # back to training mode

    # Save training generator samples
    samples_filename = os.path.join(args.output_dir, "train_samples.pkl")
    with open(samples_filename, 'wb') as f:
        pkl.dump(samples, f)
    print('Train Samples Saved')
        
    save_generator = os.path.join(args.output_dir, "trained_generator.pt")
    torch.save(G, save_generator)
    save_discriminator = os.path.join(args.output_dir, "trained_discriminator.pt")
    torch.save(D, save_discriminator)
    print('Models Trained and Saved')
    
    # finally return losses
    return losses

# call training function
losses = train(D, G, n_epochs=n_epochs)
run.log('losses', losses)
