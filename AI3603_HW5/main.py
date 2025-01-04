from model import Unet,GaussianDiffusion
from trainer import Trainer
import torch
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

if torch.cuda.is_available():
  torch.cuda.manual_seed(42)


path = './Data/'
IMG_SIZE = 64             # Size of images, do not change this if you do not know why you need to change
batch_size = 32
train_num_steps = 30000        # total training steps
lr = 0.0005
grad_steps = 1            # gradient accumulation steps, the equivalent batch size for updating equals to batch_size * grad_steps = 16 * 1
ema_decay = 0.995           # exponential moving average decay

channels = 128             # Numbers of channels of the first layer of CNN
dim_mults = (1, 2, 4, 8)        # The model size will be (channels, 2 * channels, 4 * channels, 4 * channels, 2 * channels, channels)

timesteps = 1000            # Number of steps (adding noise)
beta_schedule = 'linear'

model = Unet(
    dim = channels,
    dim_mults = dim_mults
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMG_SIZE,
    timesteps = timesteps,
    beta_schedule = beta_schedule
)

trainer = Trainer(
    diffusion,
    path,
    train_batch_size = batch_size,
    train_lr = lr,
    train_num_steps = train_num_steps,
    gradient_accumulate_every = grad_steps,
    ema_decay = ema_decay,
    save_and_sample_every = 1000,
    results_folder="./resultsCAT"
)

if __name__ == '__main__':
    print(trainer.device)

    # Train
    trainer.train()

    ckpt = './resultsCAT/model-30.pt'
    trainer.load(ckpt)
    # Random generation
    trainer.inference(output_path="./submission")
    
    # Fusion generation

    for i in range(200):
        trainer.inference2(lamda=0.5,index1=10000+2*i,index2=10001+2*i,output_path="./fusion",source_path='./source')