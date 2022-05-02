import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import glob

from torch.nn.parallel import DistributedDataParallel as DDP

from model.c2d import Resnet
from torch.utils.data import Dataset
from torchnet.dataset.splitdataset import SplitDataset
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.


class ImageDataset(Dataset):
    def __init__(self, root):
        # Transforms for low resolution images and high resolution images
        self.transform = transforms.Compose(
            [
                transforms.Resize((240,240), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.png"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)

        return {"x": img, "y": 1}

    def __len__(self):
        return len(self.files)





def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    dset = ImageDataset('data/training')

    partitions = {0:0.25,1:0.25,2:0.25,3:0.25}
    split_dset = SplitDataset(dset,partitions)
    split_dset.select(rank)
    split_loader = DataLoader(
    split_dset,
    batch_size=2,
    shuffle=True)




    # create model and move it to GPU with id rank

    model = Resnet(3,1,[1,1,1]).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for i,data in enumerate(split_loader):
        optimizer.zero_grad()
        x = data['x'].to(rank)
        y = data['y'].type(torch.float16).to(rank)
        outputs = ddp_model(x)[:,0]
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    a = mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    print(a)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    run_demo(demo_basic, world_size)