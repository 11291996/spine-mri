from .utils.model import UNetModel
from .utils.masking import mask_func
from torch.utils.data import DataLoader
import os 
import argparse
import torch
from torch import optim
from tqdm import tqdm
from ..utils import MRIDataset
from ..logger import TrainLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="1")
    parser.add_argument("--data_dir", type=str, default="/data/datasets/spine/gtu/train")
    parser.add_argument("--original_modal", type=str, default="t1")
    parser.add_argument("--target_modal", type=str, default="t2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="/data/model")
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = "cdm"
    
    subject_name = args.data_dir.split('/')[-3]
    dataset_name = args.data_dir.split('/')[-2]

    save_dir = os.path.join(args.save_dir, model_type, subject_name, dataset_name)
    if not os.path.exists(os.path.join(args.save_dir, dataset_name)):
        os.makedirs(save_dir)
    logger = TrainLogger(log_dir=save_dir, prefix=f"train_mrm_{args.original_modal}_{args.target_modal}")
    

    model = UNetModel(image_size=256, in_channels=1,
                    model_channels=96, out_channels=1, 
                    num_res_blocks=1, attention_resolutions=[32,16,8],
                    channel_mult=[1, 1, 2, 2])

    train_dataset = MRIDataset(args.data_dir, args.original_modal, args.target_modal)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=True)

    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2

    opt = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    model.train()
    model.to(device)

    num_epochs = args.num_epochs

    for epoch in tqdm(range(num_epochs), desc='Epoch', position=0):
        for i, (original_images, target_images) in enumerate(tqdm(train_loader, desc='Batches', position=1, leave=False)):
            target_images = target_images.to(device, non_blocking=True)

            opt.zero_grad()

            batch_size = original_images.shape[0]

            #masking
            masked_images, _ = mask_func(target_images, 1, 0.75, [16, 16], [16, 16])

            #diffusion
            pred_target = model(masked_images)

            loss = torch.nn.functional.mse_loss(pred_target, target_images) 

            opt.zero_grad()
            loss.backward()
            opt.step()

        logger.log(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")


    # save model
    torch.save(model.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'mrm_{args.original_modal}_{args.target_modal}.pth'))
