from .utils.mdn_model import SimpleMLP
from .utils.model import UNetModel
from torch.utils.data import DataLoader
import torch
from torch import optim
from tqdm import tqdm
import os 
import argparse
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

    mdn_model = SimpleMLP(in_channels=192, time_embed_dim = 192, model_channels=1536, bottleneck_channels=1536, out_channels=192, num_res_blocks=12)
    encoder = UNetModel(image_size=256, in_channels=1,
                    model_channels=96, out_channels=1, 
                    num_res_blocks=1, attention_resolutions=[32,16,8],
                    channel_mult=[1, 1, 2, 2])

    subject_name = args.data_dir.split("/")[-3]
    dataset_name = args.data_dir.split("/")[-2]

    save_dir = os.path.join(args.save_dir, model_type, subject_name, dataset_name)
    if not os.path.exists(os.path.join(args.save_dir, dataset_name)):
        os.makedirs(save_dir)
    logger = TrainLogger(log_dir=save_dir, prefix=f"train_mdn_{args.original_modal}_{args.target_modal}")
    
    encoder_model_dir = os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'mrm_{args.original_modal}_{args.target_modal}.pth')

    #load the encoder weights
    try:
        encoder.load_state_dict(torch.load(encoder_model_dir))
    except:
        print("Encoder model not found. Train mrm model first, or check the save directory.")
        exit()
    
    train_dataset = MRIDataset(args.data_dir, args.original_modal, args.target_modal)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=True)

    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2

    opt = optim.Adam(list(encoder.parameters()) + list(mdn_model.parameters()), lr=lr, betas=(beta1, beta2))
    encoder.eval()
    mdn_model.train()
    encoder.to(device)
    mdn_model.to(device)

    num_epochs = args.num_epochs

    for epoch in tqdm(range(num_epochs), desc='Epoch', position=0):
        for i, (original_images, target_images) in enumerate(tqdm(train_loader, desc='Batches', position=1, leave=False)):
            target_images = target_images.to(device, non_blocking=True)

            target_distribution = encoder.image_encode(target_images)

            opt.zero_grad()

            batch_size = target_images.shape[0]

            #create noise for the target images size (batch_size, 192, 32, 32)
            noise = torch.randn(batch_size, 192, device=device)
            #get time step for diffusion
            time_steps = torch.randint(1, 1000, (batch_size, ), device=device)

            #add noise to the target distribution
            alpha_t = 0.9
            noisy_target_distribution = alpha_t * target_distribution + (1 - alpha_t) * noise

            #predict the target distribution
            pred_target_distribution = mdn_model(noisy_target_distribution, time_steps)

            #use L2 loss
            loss = torch.nn.functional.mse_loss(pred_target_distribution, target_distribution)

            opt.zero_grad()
            loss.backward()
            opt.step()

        logger.log(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")

    # save model
    torch.save(model.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'mdn_{args.original_modal}_{args.target_modal}.pth'))
