import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)

    # models
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--model", type=str, default="SISN")

    # augmentations
    parser.add_argument("--use_moa", action="store_true")
    parser.add_argument("--augs", nargs="*", default=["none"])
    parser.add_argument("--prob", nargs="*", default=[1.0])
    parser.add_argument("--mix_p", nargs="*")
    parser.add_argument("--alpha", nargs="*", default=[1.0])
    parser.add_argument("--aux_prob", type=float, default=1.0)
    parser.add_argument("--aux_alpha", type=float, default=1.2)

    # dataset
    parser.add_argument("--dataset_root", type=str, default="dataset/FFHQ/1024X1024")
    parser.add_argument("--dataset", type=str, default="FSR")
    parser.add_argument("--train_val_range", type=str, default="1-850/851-950")
    parser.add_argument("--scale", type=int, default=4)
    # training setups
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decay", type=str, default="25-50-75")
    parser.add_argument("--gamma", type=int, default=0.5)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=700000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gclip", type=int, default=0)

    # misc
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--ckpt_root", type=str, default="./pt")
    parser.add_argument("--save_root", type=str, default="./output")
    
    # number of gpu
    parser.add_argument("--num_gpu", type=int, default=2)

    return parser.parse_args()


def make_template(opt):
    opt.strict_load = opt.test_only
    opt.num_groups = 10
    opt.num_blocks = 10
    opt.num_channels = 64
    opt.reduction = 16
    opt.res_scale = 1.0
    opt.max_steps = 1000
    opt.decay = "50-100-150-200-250-300-350-400"
    opt.gclip = 0.5 if opt.pretrain else opt.gclip


    # evaluation setup
    opt.crop = 6 if "FSR" in opt.dataset else 0
    opt.crop += opt.scale
    opt.eval_y_only = False


    # default augmentation policies
    if opt.use_moa:
        opt.augs = ["blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
        opt.prob = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        opt.alpha = [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        opt.aux_prob, opt.aux_alpha = 1.0, 1.2
        opt.mix_p = None


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt
