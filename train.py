from large_unet import CondRefineNetDilated as LargeNet
from original_unet import CondRefineNetDilated as OriginalNet
from trainer import NCSNTrainer
from utils import config, device, get_train_set

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--sigma_start", type=float, default=2)
    parser.add_argument("--sigma_end", type=float, default=0.1)
    parser.add_argument("--n_sigmas", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iters", type=int, default=100001)
    parser.add_argument("--model_name", type=str, default="FashionMNIST")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--logit_transform", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true", default=True)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--ngf", type=int, default=96)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--checkpoints_folder", type=str, default="checkpoints")
    parser.add_argument("--samples_folder", type=str, default="sampled_images")
    args = parser.parse_args()

    model_config = config(
        args.model_name,
        args.image_size,
        args.channels,
        args.logit_transform,
        args.random_flip,
        args.num_classes,
        args.ngf,
        args.dropout,
    )

    if args.model_name == "FashionMNIST":
        ncsn = LargeNet(model_config).to(device)
    else:
        ncsn = OriginalNet(model_config).to(device)

    dataloader = get_train_set(args.batch_size, args.model_name, args.random_flip)

    trainer = NCSNTrainer(
        ncsn,
        lr=args.lr,
        dataloader=dataloader,
        num_iters=args.num_iters,
        beta1=args.beta1,
        beta2=args.beta2,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        n_sigmas=args.n_sigmas,
        checkpoints_folder=args.checkpoints_folder,
        save_every=args.save_every,
        weight_decay=args.weight_decay,
        ema_rate=args.ema_rate,
        checkpoint_name_prefix=args.model_name,
    )
    trainer.train_ncsn()
