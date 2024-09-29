from os import path
from os.path import join as pj
import time
import random
import argparse

from tqdm import tqdm
import math
import numpy as np
import torch as th
from torch import nn, autograd
import matplotlib.pyplot as plt
import umap
import re

from modules import models, utils
from modules.datasets import MultimodalDataset
from modules.datasets import MultiDatasetSampler


parser = argparse.ArgumentParser()
## Task
parser.add_argument('--task', type=str, default='wnn_rna',
    help="Choose a task")
parser.add_argument('--reference', type=str, default='',
    help="Choose a reference task")
parser.add_argument('--experiment', type=str, default='e0',
    help="Choose an experiment")
parser.add_argument('--model', type=str, default='default',
    help="Choose a model configuration")
# parser.add_argument('--data', type=str, default='sup',
#     help="Choose a data configuration")
parser.add_argument('--actions', type=str, nargs='+', default=['train'],
    help="Choose actions to run")
parser.add_argument('--method', type=str, default='midas',
    help="Choose an method to benchmark")
parser.add_argument('--init_model', type=str, default='',
    help="Load a trained model")
parser.add_argument('--init_from_ref', type=int, default=0,
    help="Load a model trained on the reference task")
parser.add_argument('--mods_conditioned', type=str, nargs='+', default=[],
    help="Modalities conditioned for sampling")
parser.add_argument('--data_conditioned', type=str, default='prior.csv',
    help="Data conditioned for sampling")
parser.add_argument('--sample_num', type=int, default=0,
    help='Number of samples to be generated')
parser.add_argument('--input_mods', type=str, nargs='+', default=[],
    help="Input modalities for transformation")
## Training
parser.add_argument('--epoch_num', type=int, default=2000,
    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4,
    help='Learning rate')
parser.add_argument('--grad_clip', type=float, default=-1,
    help='Gradient clipping value')
parser.add_argument('--s_drop_rate', type=float, default=0.1,
    help="Probility of dropping out subject ID during training")
parser.add_argument('--seed', type=int, default=3407,
    help="Set the random seed to reproduce the results")
parser.add_argument('--use_shm', type=int, default=1,
    help="Use shared memory to accelerate training")
## Debugging
parser.add_argument('--print_iters', type=int, default=-1,
    help="Iterations to print training messages")
parser.add_argument('--log_epochs', type=int, default=100,
    help='Epochs to log the training states')
parser.add_argument('--save_epochs', type=int, default=10,
    help='Epochs to save the latest training states (overwrite previous ones)')
parser.add_argument('--time', type=int, default=0, choices=[0, 1],
    help='Time the forward and backward passes')
parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
    help='Print intermediate variables')
# o, _ = parser.parse_known_args()  # for python interactive
o = parser.parse_args()


# Initialize global varibles
data_config = None
net = None
discriminator = None
optimizer_net = None
optimizer_disc = None
benchmark = {
    "train_loss": [],
    "test_loss": [],
    "foscttm": [],
    "epoch_id_start": 0
}


def main():
    initialize()
    if o.actions == "print_model":
        print_model()
    if "train" in o.actions:
        train()
    if "test" in o.actions:
        test()
    if "save_input" in o.actions:
        predict(joint_latent=False, input=True)
    if "predict_all" in o.actions:
        predict(mod_latent=True, impute=True, batch_correct=True, translate=True, input=True)
    if "predict_joint" in o.actions:
        predict()
    if "predict_all_latent" in o.actions:
        predict(mod_latent=True)
    if "impute" in o.actions:
        predict(impute=True, input=True)
    if "translate" in o.actions:
        predict(translate=True, input=True)
    if "batch_correct" in o.actions:
        predict(batch_correct=True, input=True)
    if "predict_all_latent_bc" in o.actions:
        predict(mod_latent=True, batch_correct=True, input=True)

    if "visualize" in o.actions:
        visualize()
    if "benchmark" in o.actions:
        z, c, b, subset_ids, batch_ids = utils.load_predicted(o, o.result_dir)
        calc_metrics(z, c, batch_ids)
        



def initialize():
    init_seed()
    init_dirs()
    load_data_config()
    load_model_config()
    get_gpu_config()
    init_model()


def init_seed():
    if o.seed >= 0:
        np.random.seed(o.seed)
        th.manual_seed(o.seed)
        th.cuda.manual_seed_all(o.seed)


def init_dirs():
    data_folder = re.sub("_generalize", "_transfer", o.task)
    if o.use_shm == 1:
        o.data_dir = pj("/root/data/DBP_sa_bc/data", "processed", data_folder)
        # o.data_dir = pj("/root/lry/DBP_sa_bc/data", "processed", data_folder)
    else:
        o.data_dir = pj("data", "processed", data_folder)
    o.result_dir = pj("result", o.task, o.experiment, o.model)
    o.pred_dir = pj(o.result_dir, "predict", o.init_model)
    o.train_dir = pj(o.result_dir, "train")
    o.debug_dir = pj(o.result_dir, "debug")
    utils.mkdirs([o.train_dir, o.debug_dir])
    print("Task: %s\nExperiment: %s\nModel: %s\n" % (o.task, o.experiment, o.model))


def load_data_config():
    get_dims_x()
    o.mods = list(o.dims_x.keys())
    o.mod_num = len(o.dims_x)
    
    global data_config
    cfg_task = re.sub("_atlas|_generalize|_transfer|_ref_.*", "", o.task)
    data_config = utils.load_toml("/root/data/DBP_sa_bc/configs/data.toml")[cfg_task]
    # data_config = utils.load_toml("/root/lry/DBP_sa_bc/configs/data.toml")[cfg_task]
    for k, v in data_config.items():
        vars(o)[k] = v

    o.s_joint, o.combs, o.s, o.dims_s = utils.gen_all_batch_ids(o.s_joint, o.combs)
    
    if o.reference != '':

        o.dims_s = {k: v + 1 for k, v in o.dims_s.items()}

        data_config_ref = utils.load_toml("/root/data/DBP_sa_bc/configs/data.toml")[o.reference]
        # data_config_ref = utils.load_toml("/root/lry/DBP_sa_bc/configs/data.toml")[o.reference]
        _, _, s_ref, dims_s_ref = utils.gen_all_batch_ids(data_config_ref["s_joint"], 
                                                    data_config_ref["combs"])
        o.subset_ids_ref = {m: [] for m in dims_s_ref}
        for subset_id, id_dict in enumerate(s_ref):
            for m in id_dict.keys():
                o.subset_ids_ref[m].append(subset_id)

    o.dim_s = o.dims_s["joint"]
    o.dim_b = 2



def load_model_config():
    model_config = utils.load_toml("/root/data/DBP_sa_bc/configs/model.toml")["default"]
    # model_config = utils.load_toml("/root/lry/DBP_sa_bc/configs/model.toml")["default"]
    if o.model != "default":
        model_config.update(utils.load_toml("/root/data/DBP_sa_bc/configs/model.toml")[o.model])
        # model_config.update(utils.load_toml("/root/lry/DBP_sa_bc/configs/model.toml")[o.model])
    for k, v in model_config.items():
        vars(o)[k] = v
    o.dim_z = o.dim_c + o.dim_b
    o.dims_dec_x = o.dims_enc_x[::-1]
    o.dims_dec_s = o.dims_enc_s[::-1]
    if "dims_enc_chr" in vars(o).keys():
        o.dims_dec_chr = o.dims_enc_chr[::-1]
    o.dims_h = {}
    for m, dim in o.dims_x.items():
        o.dims_h[m] = dim if m != "atac" else o.dims_enc_chr[-1] * 22


def get_gpu_config():
    o.G = 1  # th.cuda.device_count()  # get GPU number
    assert o.N % o.G == 0, "Please ensure the mini-batch size can be divided " \
        "by the GPU number"
    o.n = o.N // o.G
    print("Total mini-batch size: %d, GPU number: %d, GPU mini-batch size: %d" % (o.N, o.G, o.n))


def init_model():
    """
    Initialize the model, optimizer, and benchmark
    """
    global net, discriminator, optimizer_net, optimizer_disc
    
    # Initialize models
    net = models.Net(o).cuda()
    discriminator = models.Discriminator(o).cuda()
    net_param_num = sum([param.data.numel() for param in net.parameters()])
    disc_param_num = sum([param.data.numel() for param in discriminator.parameters()])
    print('Parameter number: %.3f M' % ((net_param_num+disc_param_num) / 1e6))
    
    # Load benchmark
    if o.init_model != '':
        if o.init_from_ref == 0:
            fpath = pj(o.train_dir, o.init_model)
            savepoint_toml = utils.load_toml(fpath+".toml")
            benchmark.update(savepoint_toml['benchmark'])
            o.ref_epoch_num = savepoint_toml["o"]["ref_epoch_num"]
        else:
            fpath = pj("result", o.reference, o.experiment, o.model, "train", o.init_model)
            benchmark.update(utils.load_toml(fpath+".toml")['benchmark'])
            o.ref_epoch_num = benchmark["epoch_id_start"]
    else:
        o.ref_epoch_num = 0

    # Initialize optimizers
    optimizer_net = th.optim.AdamW(net.parameters(), lr=o.lr)
    optimizer_disc = th.optim.AdamW(discriminator.parameters(), lr=o.lr)
    
    # Load models and optimizers
    if o.init_model != '':
        savepoint = th.load(fpath+".pt")
        if o.init_from_ref == 0:
            net.load_state_dict(savepoint['net_states'])
            discriminator.load_state_dict(savepoint['disc_states'])
            optimizer_net.load_state_dict(savepoint['optim_net_states'])
            optimizer_disc.load_state_dict(savepoint['optim_disc_states'])
        else:
            exclude_modules = ["s_enc", "s_dec"]
            pretrained_dict = {}
            for k, v in savepoint['net_states'].items():
                exclude = False
                for exclude_module in exclude_modules:
                    if exclude_module in k:
                        exclude = True
                        break
                if not exclude:
                    pretrained_dict[k] = v
            net_dict = net.state_dict()
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
        print('Model is initialized from ' + fpath + ".pt")


def print_model():
    global net, discriminator
    with open(pj(o.result_dir, "model_architecture.txt"), 'w') as f:
        print(net, file=f)
        print(discriminator, file=f)


def get_dims_x():
    dims_x = utils.load_csv(pj(o.data_dir, "feat", "feat_dims.csv"))
    # mods = dims_x[0][1:]
    # dims = list(map(int, dims_x[1][1:]))
    # o.dims_x = utils.get_dict(mods, dims)
    dims_x = utils.transpose_list(dims_x)
    o.dims_x = {}
    for i in range(1, len(dims_x)):
        m = dims_x[i][0]
        if m == "atac":
            o.dims_chr = list(map(int, dims_x[i][1:]))
            o.dims_x[m] = sum(o.dims_chr)
        else:
            o.dims_x[m] = int(dims_x[i][1])


    # in_dirs = {}
    # for m in o.mods:
    #     in_dirs[m] = pj(o.data_dir, m)
    # filenames = utils.get_filenames(in_dirs[o.mods[0]], "csv")
    # o.dims_x = {}
    # for m in o.mods:
    #     if m == "label":
    #         o.dims_x[m] = o.class_num
    #     else:
    #         file_path = pj(in_dirs[m], filenames[0])
    #         v = utils.load_csv(file_path)
    #         o.dims_x[m] = len(v[0])

    print("Input feature numbers: ", o.dims_x)


def train():
    # train_data_loaders = get_dataloaders("train")
    # test_data_loaders = get_dataloaders("test")
    train_data_loader_cat = get_dataloader_cat("train", train_ratio=None)
    # test_data_loader_cat = get_dataloader_cat("test", train_ratio=None)
    for epoch_id in range(benchmark['epoch_id_start'], o.epoch_num):
        run_epoch(train_data_loader_cat, "train", epoch_id)
        # run_epoch(train_data_loaders, "train", epoch_id)
        # run_epoch(test_data_loader_cat, "test", epoch_id)
        # run_epoch(test_data_loaders, "test", epoch_id)
        # utils.calc_subsets_foscttm(net.sct, test_data_loaders, benchmark["foscttm"], "test", 
        #                            o.epoch_num, epoch_id)
        check_to_save(epoch_id)


def get_dataloaders(split, train_ratio=None):
    data_loaders = {}
    for subset in range(len(o.s)):
        data_loaders[subset] = get_dataloader(subset, split, train_ratio=train_ratio)
    return data_loaders


def get_dataloader(subset, split, train_ratio=None):
    dataset = MultimodalDataset(o.task, o.data_dir, subset, split, train_ratio=train_ratio)
    shuffle = True if split == "train" else False
    data_loader = th.utils.data.DataLoader(dataset, batch_size=o.N, shuffle=shuffle,
                                           num_workers=64, pin_memory=True)
    print("Subset: %d, modalities %s: %s size: %d" %
          (subset, str(o.combs[subset]), split, dataset.size))
    return data_loader


def get_dataloader_cat(split, train_ratio=None):
    datasets = []
    for subset in range(len(o.s)):
        datasets.append(MultimodalDataset(o.task, o.data_dir, subset, split, train_ratio=train_ratio))
        print("Subset: %d, modalities %s: %s size: %d" %  (subset, str(o.combs[subset]), split,
            datasets[subset].size))
    dataset_cat = th.utils.data.dataset.ConcatDataset(datasets)
    shuffle = True if split == "train" else False
    sampler = MultiDatasetSampler(dataset_cat, batch_size=o.N, shuffle=shuffle)
    data_loader = th.utils.data.DataLoader(dataset_cat, batch_size=o.N, sampler=sampler, 
        num_workers=64, pin_memory=True)
    return data_loader


def test():
    data_loaders = get_dataloaders()
    run_epoch(data_loaders, "test")


def run_epoch(data_loader, split, epoch_id=0):
    if split == "train":
        net.train()
        discriminator.train()
    elif split == "test":
        net.eval()
        discriminator.eval()
    else:
        assert False, "Invalid split: %s" % split

    loss_total = 0
    for i, data in enumerate(data_loader):
        loss = run_iter(split, epoch_id, data)
        loss_total += loss
        if o.print_iters > 0 and (i+1) % o.print_iters == 0:
            print('%s\tepoch: %d/%d\tBatch: %d/%d\t%s_loss: %.3f'.expandtabs(3) % 
                  (o.task, epoch_id+1, o.epoch_num, i+1, len(data_loader), split, loss))
    loss_avg = loss_total / len(data_loader)
    print('%s\tepoch: %d/%d\t%s_loss: %.3f\n'.expandtabs(3) % 
          (o.task, epoch_id+1, o.epoch_num, split, loss_avg))
    benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
    return loss_avg


# # Train by iterating over subsets
# def run_epoch(data_loaders, split, epoch_id=0):
#     if split == "train":
#         net.train()
#         discriminator.train()
#     elif split == "test":
#         net.eval()
#         discriminator.eval()
#     else:
#         assert False, "Invalid split: %s" % split
#     losses = {}
#     for subset, data_loader in data_loaders.items():
#         losses[subset] = 0
#         for i, data in enumerate(data_loader):
#             loss = run_iter(split, epoch_id, data)
#             losses[subset] += loss
#             if o.print_iters > 0 and (i+1) % o.print_iters == 0:
#                 print('Epoch: %d/%d, subset: %s, Batch: %d/%d, %s loss: %.3f' % (epoch_id+1,
#                 o.epoch_num, str(subset), i+1, len(data_loader), split, loss))
#         losses[subset] /= len(data_loader)
#         print('Epoch: %d/%d, subset: %d, %s loss: %.3f' % (epoch_id+1, o.epoch_num, subset, 
#             split, losses[subset]))
#     loss_avg = sum(losses.values()) / len(losses.keys())
#     print('Epoch: %d/%d, %s loss: %.3f\n' % (epoch_id+1, o.epoch_num, split, loss_avg))
#     benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
#     return loss_avg


def run_iter(split, epoch_id, inputs):
    inputs = utils.convert_tensors_to_cuda(inputs)
    if split == "train":
        with autograd.set_detect_anomaly(o.debug == 1):
            loss_net, cw_all = forward_net(inputs)
            discriminator.epoch = epoch_id - o.ref_epoch_num
            K = 3
            for _ in range(K):
                loss_disc = forward_disc(utils.detach_tensors(cw_all), inputs["s"])
                update_disc(loss_disc)
            # c = models.CheckBP('c')(c)
            loss_adv = forward_disc(cw_all, inputs["s"])
            loss_adv = -loss_adv
            loss = loss_net + loss_adv
            update_net(loss)
            
            print("loss_net: %.3f\tloss_adv: %.3f\tprob: %.4f".expandtabs(3) %
                  (loss_net.item(), loss_adv.item(), discriminator.prob))
            
        # Additionally train the discriminator after the last training subset
        if o.reference != '' and inputs["s"]["joint"][0, 0].item() == o.dim_s - 2:
            
            # Randomly load c inferred from the reference dataset
            c_all_ref = {}
            subset_ids_sampled = {m: random.choice(ids) for m, ids in o.subset_ids_ref.items()}
            for m, subset_id in subset_ids_sampled.items():
                z_dir = pj("result", o.reference, o.experiment, o.model, "predict", o.init_model,
                        "subset_"+str(subset_id), "z", m)
                filename = random.choice(utils.get_filenames(z_dir, "csv"))
                z = th.from_numpy(np.array(utils.load_csv(pj(z_dir, filename)), dtype=np.float32))
                # z = th.tensor(utils.load_csv(pj(z_dir, filename)), dtype=th.float32)
                c_all_ref[m] = z[:, :o.dim_c]
            c_all_ref = utils.convert_tensors_to_cuda(c_all_ref)

            # Generate s for the reference dataset, which is treated as the last subset
            s_ref = {}
            tmp = inputs["s"]["joint"]
            for m, d in o.dims_s.items():
                s_ref[m] = th.full((c_all_ref[m].size(0), 1), d-1, dtype=tmp.dtype, device=tmp.device)

            with autograd.set_detect_anomaly(o.debug == 1):
                for _ in range(K):
                    loss_disc = forward_disc(c_all_ref, s_ref)
                    update_disc(loss_disc)
            
    else:
        with th.no_grad():
            loss_net, cw_all = forward_net(inputs)
            loss_adv = forward_disc(cw_all, inputs["s"])
            loss_adv = -loss_adv
            loss = loss_net + loss_adv
            
            # print("loss_net: %.3f\tloss_adv: %.3f\tprob: %.4f".expandtabs(3) %
            #       (loss_net.item(), loss_adv.item(), discriminator.prob))
            
    # if o.time == 1:
    #     print('Runtime: %.3fs' % (forward_time + backward_time))
    return loss.item()


def forward_net(inputs):
    return net(inputs)


def forward_disc(c, s):
    return discriminator(c, s)


def update_net(loss):
    update(loss, net, optimizer_net)


def update_disc(loss):
    update(loss, discriminator, optimizer_disc)
    

def update(loss, model, optimizer):
    optimizer.zero_grad()
    loss.backward()
    if o.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
    optimizer.step()


def check_to_save(epoch_id):
    if (epoch_id+1) % o.log_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_%08d" % epoch_id)
    if (epoch_id+1) % o.save_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_latest")


def save_training_states(epoch_id, filename):
    benchmark['epoch_id_start'] = epoch_id + 1
    utils.save_toml({"o": vars(o), "benchmark": benchmark}, pj(o.train_dir, filename+".toml"))
    th.save({"net_states": net.state_dict(),
             "disc_states": discriminator.state_dict(),
             "optim_net_states": optimizer_net.state_dict(),
             "optim_disc_states": optimizer_disc.state_dict()
            }, pj(o.train_dir, filename+".pt"))


def predict(joint_latent=True, mod_latent=False, impute=False, batch_correct=False, translate=False, 
            input=False):
    if translate:
        mod_latent = True
    print("Predicting ...")
    dirs = utils.get_pred_dirs(o, joint_latent, mod_latent, impute, batch_correct, translate, input)
    utils.mkdirs(dirs, remove_old=True)
    data_loaders = get_dataloaders("test", train_ratio=0)
    net.eval()
    with th.no_grad():
        for subset_id, data_loader in data_loaders.items():
            print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
            
            for i, data in enumerate(tqdm(data_loader)):
                data = utils.convert_tensors_to_cuda(data)
                
                # conditioned on all observed modalities
                if joint_latent:
                    x_r_pre, _, _, _, z, _, _, _, _, A, _, _, _, _, _, w, A_mu, A_logvar, EW, EA = net.sct(data)  # N * K
                    utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"]["joint"], fname_fmt) % i)
                    utils.save_tensor_to_csv(A, pj(dirs[subset_id]["A"]["joint"], fname_fmt))
                    utils.save_tensor_to_csv(w, pj(dirs[subset_id]["w"]["joint"], fname_fmt) % i)
                    utils.save_tensor_to_csv(EW, pj(dirs[subset_id]["EW"]["joint"], fname_fmt) % i)
                    utils.save_tensor_to_csv(EA, pj(dirs[subset_id]["EA"]["joint"], fname_fmt))
                if impute:
                    x_r = models.gen_real_data(x_r_pre, sampling=False)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_impt"][m], fname_fmt) % i)
                if input:  # save the input
                    for m in o.combs[subset_id]:
                        utils.save_tensor_to_csv(data["x"][m].int(), pj(dirs[subset_id]["x"][m], fname_fmt) % i)

                # conditioned on each individual modalities
                if mod_latent:
                    for m in data["x"].keys():
                        input_data = {
                            "x": {m: data["x"][m]},
                            "s": data["s"], 
                            "e": {}
                        }
                        if m in data["e"].keys():
                            input_data["e"][m] = data["e"][m]
                        x_r_pre, _, _, _, z, _, _, _, _, A, _, _, _, _, _, w, A_mu, A_logvar, EW, EA = net.sct(input_data)  # N * K
                        utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"][m], fname_fmt) % i)
                        if translate:
                            x_r = models.gen_real_data(x_r_pre, sampling=False)
                            for m_ in set(o.mods) - {m}:
                                utils.save_tensor_to_csv(x_r[m_], pj(dirs[subset_id]["x_trans"][m+"_to_"+m_], fname_fmt) % i)

        if batch_correct:
            print("Calculating b_centroid ...")
            # z, c, b, subset_ids, batch_ids = utils.load_predicted(o)
            # b = th.from_numpy(b["joint"])
            # subset_ids = th.from_numpy(subset_ids["joint"])
            
            pred = utils.load_predicted(o)
            b = th.from_numpy(pred["z"]["joint"][:, o.dim_c:])
            s = th.from_numpy(pred["s"]["joint"])

            b_mean = b.mean(dim=0, keepdim=True)
            b_subset_mean_list = []
            for subset_id in s.unique():
                b_subset = b[s == subset_id, :]
                b_subset_mean_list.append(b_subset.mean(dim=0))
            b_subset_mean_stack = th.stack(b_subset_mean_list, dim=0)
            dist = ((b_subset_mean_stack - b_mean) ** 2).sum(dim=1)
            net.sct.b_centroid = b_subset_mean_list[dist.argmin()]
            net.sct.batch_correction = True
            
            print("Batch correction ...")
            for subset_id, data_loader in data_loaders.items():
                print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
                fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
                
                for i, data in enumerate(tqdm(data_loader)):
                    data = utils.convert_tensors_to_cuda(data)
                    x_r_pre, *_ = net.sct(data)
                    x_r = models.gen_real_data(x_r_pre, sampling=True)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_bc"][m], fname_fmt) % i)
        
            

# def calc_b_centroid():
#     print("Calculating b_centroid ...")
#     predict()
#     z, c, b, subset_ids, batch_ids = utils.load_predicted(o)
#     b = th.from_numpy(b["joint"])
#     subset_ids = th.from_numpy(subset_ids["joint"])
#     b_mean = b.mean(dim=0, keepdim=True)
#     b_subset_mean_list = []
#     for subset_id in subset_ids.unique():
#         b_subset = b[subset_ids == subset_id, :]
#         b_subset_mean_list.append(b_subset.mean(dim=0))
#     b_subset_mean_stack = th.stack(b_subset_mean_list, dim=0)
#     dist = ((b_subset_mean_stack - b_mean) ** 2).sum(dim=1)
#     b_centroid = b_subset_mean_list[dist.argmin()]
#     return b_centroid


# def batch_correct():
#     print("Batch correction ...")
#     b_centroid = calc_b_centroid()
#     dirs = {}
#     data_loaders = get_dataloaders("test", train_ratio=0)
#     net.eval()
#     net.sct.batch_correction = True
#     net.sct.b_centroid = b_centroid
#     with th.no_grad():
#         for subset_id, data_loader in data_loaders.items():
#             print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            
#             dirs[subset_id] = {"x_bc": {}}
#             for m in o.combs[subset_id]:
#                 dirs[subset_id]["x_bc"][m] = pj(o.pred_dir, "subset_"+str(subset_id), "x_bc", m)
#                 utils.mkdirs(dirs[subset_id]["x_bc"][m], remove_old=True)
#             fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
            
#             for i, data in enumerate(tqdm(data_loader)):
#                 data = utils.convert_tensors_to_cuda(data)
                
#                 # conditioned on all observed modalities
#                 x_r_pre, *_ = net.sct(data)  # N * K
#                 x_bc = models.gen_real_data(x_r_pre, sampling=True)
#                 for m in o.combs[subset_id]:
#                     utils.save_tensor_to_csv(x_bc[m], pj(dirs[subset_id]["x_bc"][m], fname_fmt) % i)


def calc_foscttm():
    data_loaders = get_dataloaders("test", train_ratio=0)
    utils.calc_subsets_foscttm(o, net.sct, data_loaders, benchmark["foscttm"], "test")


def visualize():
    pred = utils.load_predicted(o, mod_latent=True)
    z = pred["z"]
    s = pred["s"]
            
    print("Computing UMAP ...")
    umaps = {}
    mods = o.mods + ["joint"]
    for m in mods:
        print("Computing UMAP: " + m)
        umaps[m] = {
            "z": umap.UMAP(n_neighbors=100, random_state=42).fit_transform(z[m]),
            "c": umap.UMAP(n_neighbors=100, random_state=42).fit_transform(z[m][:, :o.dim_c]),
            "b": umap.UMAP(n_neighbors=100, random_state=42).fit_transform(z[m][:, o.dim_c:])
        }

    print("Plotting ...")
    fig_rows = len(o.s_joint) + 1
    fig_fcols = len(mods) * 3
    fsize = 4
    pt_size = [0.03]
    fg_color = [(0.7, 0, 0)]
    bg_color = [(0.3, 1, 1)]
    fig, ax = plt.subplots(fig_rows, fig_fcols, figsize=(fig_fcols*fsize, fig_rows*fsize))

    for i, m in enumerate(umaps.keys()):
        for j, d in enumerate(umaps[m].keys()):
            col = i * 3 + j
            v = umaps[m][d]

            # for each subset
            for subset_id, batch_id in enumerate(o.s_joint):
                fg = (s[m]==subset_id)
                if sum(fg) > 0:
                    bg = ~fg
                    ax[subset_id, col].scatter(v[bg, 0], v[bg, 1], label=m, s=pt_size, 
                        c=bg_color, marker='.')
                    ax[subset_id, col].scatter(v[fg, 0], v[fg, 1], label=m, s=pt_size, 
                        c=fg_color, marker='.')
                    ax[subset_id, col].set_title("subset_%s, batch_%s, %s, %s"
                                                 % (subset_id, batch_id, m, d))

            # for all subsets
            ax[fig_rows-1, col].scatter(v[:, 0], v[:, 1], label=m, s=pt_size, 
                c=fg_color, marker='.')
            ax[fig_rows-1, col].set_title("subset_all, batch_all, %s, %s" % (m, d))

    plt.tight_layout()
    fig_dir = pj(o.result_dir, "predict", o.init_model, "fig")
    utils.mkdirs(fig_dir, remove_old=True)
    plt.savefig(pj(fig_dir, "predict_all.png"))
    # plt.show()


def calc_metrics(z, c, batch_ids):

    import scib
    import scib.metrics as me
    import anndata as ad
    import scanpy as sc
    
    embed = "X_emb"
    batch_key = "batch"
    label_key = "label"
    cluster_key = "cluster"
    si_metric = "euclidean"
    subsample = 0.5
    verbose = False

    labels = []
    for raw_data_dir in o.raw_data_dirs:
        label = utils.load_csv(pj(raw_data_dir, "label_seurat", "l1.csv"))
        # label = utils.load_csv(pj(raw_data_dir, "label", "main.csv"))
        labels += utils.transpose_list(label)[1][1:]
    # print(len(np.unique(labels)))

    if o.method in ["midas"]:
        output_type = "embed"
    elif o.method in ["harmony_wnn", "unintg_wnn"]:
        output_type = "graph"
    else:
        output_type = None

    adata = ad.AnnData(z["joint"])
    adata.obsm[embed] = z["joint"]
    adata.obs[batch_key] = batch_ids["joint"].astype(str)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[label_key] = labels
    adata.obs[label_key] = adata.obs[label_key].astype("category")

    import scipy
    if o.method == "midas":
        adata_int = ad.AnnData(c["joint"])
        adata_int.obsm[embed] = c["joint"]
        adata_int.obs[batch_key] = batch_ids["joint"].astype(str)
        adata_int.obs[batch_key] = adata_int.obs[batch_key].astype("category")
        adata_int.obs[label_key] = labels
        adata_int.obs[label_key] = adata_int.obs[label_key].astype("category")
    elif o.method == "harmony_wnn":
        adata_int = ad.AnnData(c["joint"]*0)
        adata_int.obs[batch_key] = batch_ids["joint"].astype(str)
        adata_int.obs[batch_key] = adata_int.obs[batch_key].astype("category")
        adata_int.obs[label_key] = labels
        adata_int.obs[label_key] = adata_int.obs[label_key].astype("category")
        adata_int.obsp["connectivities"] = \
            scipy.io.mmread("result/comparison/"+o.task+"/harmony+wnn/connectivities.mtx").tocsr()
        adata_int.uns["neighbors"] = {'connectivities_key': 'connectivities'}
    elif o.method == "unintg_wnn":
        adata_int = ad.AnnData(c["joint"]*0)
        adata_int.obs[batch_key] = batch_ids["joint"].astype(str)
        adata_int.obs[batch_key] = adata_int.obs[batch_key].astype("category")
        adata_int.obs[label_key] = labels
        adata_int.obs[label_key] = adata_int.obs[label_key].astype("category")
        adata_int.obsp["connectivities"] = \
            scipy.io.mmread("result/comparison/"+o.task+"/unintg+wnn/connectivities.mtx").tocsr()
        adata_int.uns["neighbors"] = {'connectivities_key': 'connectivities'}
    else:
        assert False, "Invalid method!"

    results = {}

    print('clustering...')
    res_max, nmi_max, nmi_all = scib.clustering.opt_louvain(adata_int, label_key=label_key,
        cluster_key=cluster_key, function=me.nmi, plot=False, verbose=verbose, inplace=True)

    print('NMI...')
    results['NMI'] = me.nmi(adata_int, group1=cluster_key, group2=label_key, method='arithmetic')

    print('ARI...')
    results['ARI'] = me.ari(adata_int, group1=cluster_key, group2=label_key)

    if output_type == "embed":
        print('asw label...')
        results['ASW_label'] = me.silhouette(adata_int, group_key=label_key, embed=embed,
            metric=si_metric)
    
        print('asw batch...')
        results['ASW_batch'] = me.silhouette_batch(adata_int, batch_key=batch_key,
            group_key=label_key, embed=embed, metric=si_metric, verbose=verbose)
    
        print('PC regression...')
        results['PCR_batch'] = me.pcr_comparison(adata, adata_int, embed=embed, covariate=batch_key,
            verbose=verbose)
        
        print("isolated score asw...")
        results['il_score_asw'] = me.isolated_labels(adata_int, label_key=label_key, 
            batch_key=batch_key, embed=embed, cluster=False, verbose=verbose) 

        print('kBET...')
        results['kBET'] = me.kBET(adata_int, batch_key=batch_key, label_key=label_key, embed=embed, 
            verbose=verbose)

    elif output_type == "graph":
        results['ASW_label'] = np.nan
        results['ASW_batch'] = np.nan
        results['PCR_batch'] = np.nan
        results['il_score_asw'] = np.nan

        print('kBET...')
        results['kBET'] = me.kBET(adata_int, batch_key=batch_key, label_key=label_key, embed=embed, 
            type_="knn", verbose=verbose)

    print("isolated score f1...")
    results['il_score_f1'] = me.isolated_labels(adata_int, label_key=label_key, batch_key=batch_key,
        embed=embed, cluster=True, verbose=verbose)

    print('Graph connectivity...')
    results['graph_conn'] = me.graph_connectivity(adata_int, label_key=label_key)

    print('cLISI score...')
    results['cLISI'] = me.clisi_graph(adata_int, batch_key=batch_key, label_key=label_key, 
        subsample=subsample*100, n_cores=1, verbose=verbose)

    print('iLISI score...')
    results['iLISI'] = me.ilisi_graph(adata_int, batch_key=batch_key, subsample=subsample*100,
        n_cores=1, verbose=verbose)

    results = {k: float(v) for k, v in results.items()}
    batch_score = np.nanmean([
        results['PCR_batch'], 
        results['ASW_batch'],
        results['iLISI'], 
        results['graph_conn'], 
        results['kBET']])
    bio_score = np.nanmean([
        results['NMI'], 
        results['ARI'], 
        results['ASW_label'], 
        results['il_score_f1'], 
        results['il_score_asw'], 
        results['cLISI']])
    results["final"] = float(0.4 * batch_score + 0.6 * bio_score)

    metric_dir = pj(o.result_dir, "predict", o.init_model, "metric")
    utils.mkdirs(metric_dir, remove_old=False)
    utils.save_toml(results, pj(metric_dir, "metrics_"+o.method+".toml"))

main()
