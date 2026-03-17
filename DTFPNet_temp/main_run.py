import argparse
import datetime
import math
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
import numpy as np
import torch.optim as optim
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score
from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool

from math import ceil
from layer import *
from torch_dct import dct, idct

import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
class DCT(nn.Module):
    def __init__(self, len):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(len, dtype=torch.float32))
        trunc_normal_(self.complex_weight, std=.02)

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply DCT along the time dimension
        dct_data = dct(x)
        x_weighted = dct_data * self.complex_weight

        # Apply Inverse DCT
        x = idct(x_weighted)
        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape
        return x
class DCT_GRU(nn.Module):
    def __init__(self, dim, GRU_layers):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, dtype=torch.float32))
        self.gru_in = nn.GRU(input_size=dim, hidden_size=dim, num_layers=GRU_layers, batch_first=True)

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply DCT along the time dimension
        dct_data = dct(x)

        dct_data = dct_data.transpose(1, 2)
        dct_data,hi_ = self.gru_in(dct_data)
        dct_data = dct_data.transpose(1, 2)

        # Apply Inverse DCT
        x = idct(dct_data)
        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape
        return x
class DFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.dim = dim

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')  
        weight = torch.view_as_complex(self.complex_weight)  
        x_weighted = x_fft * weight

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=self.dim, dim=-1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x
class GNNStack(nn.Module):
    """ The stack layers of GNN.
    """
    def __init__(self):
        super().__init__()
        # TODO: Sparsity Analysis
        gnn_model_type = args.arch
        num_layers = args.num_layers
        groups = args.groups
        kern_size = args.kern_size
        kern_size_mid = args.kern_size_mid
        in_dim = args.in_dim
        hidden_dim = args.hidden_dim
        out_dim = args.out_dim
        seq_len = args.seq_len

        num_nodes = args.num_channels
        num_classes = args.num_classes

        k_neighs = self.num_nodes = num_nodes
        self.pos_drop = nn.Dropout(p=args.dropout_rate)
        self.num_graphs = args.groups
        self.groups=groups
        self.seq_mid_len = 0
        self.num_feats = seq_len
        if seq_len % groups:
            self.num_feats += (groups - seq_len % groups)
        self.g_constr = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)  # return adj
        self.g_constr_fre = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)  # return adj
        if seq_len % self.num_graphs:
            pad_size = (self.num_graphs - seq_len % self.num_graphs) / 2
            temp_length = F.pad(torch.randn(3,3,4,2,seq_len), (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
            self.seq_len_new = temp_length.size(-1)
        else:
            self.seq_len_new = seq_len
        GRU_layers = args.GRU_layers

        self.asb = DCT_GRU(dim=num_nodes,GRU_layers=GRU_layers)
        self.asb_1 = DCT_GRU(dim=num_nodes,GRU_layers=GRU_layers)
        # self.asb = DCT(len=self.seq_len_new)
        # self.asb_1 = DCT(len=self.seq_len_new)
        # self.asb = DFT(dim=self.seq_len_new)
        # self.asb_1 = DFT(dim=self.seq_len_new)

        gnn_model, heads = self.build_gnn_model(gnn_model_type)  # gnn_model=DenseGINConv2d  heads=1

        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'

        # To make the framework presented in this paper more accessible, the code below are deliberately kept concise and avoid loop constructs. 
        # Furthermore, as superior performance compared to SoTA methods was already achieved using only three layers, 
        # we did not investigate whether stacking deeper architectures could yield further performance improvements.
        self.tconvs_1 = nn.Conv2d(1, in_dim, (1, kern_size[0]), padding="same")
        self.gconvs_1 = gnn_model(in_dim, heads * in_dim, groups)
        self.bns_1 = nn.BatchNorm2d(heads * in_dim)

        self.tconvs_tmp_1 = nn.Conv2d(in_dim*2, in_dim, (1, kern_size_mid[0]), padding="same")
        self.tconvs_tmp_2 = nn.Conv2d(hidden_dim*2, hidden_dim, (1, kern_size_mid[1]), padding="same")
        self.tconvs_tmp_3 = nn.Conv2d(out_dim*2, out_dim, (1, kern_size_mid[2]), padding="same")

        self.tconvs_2 = nn.Conv2d(in_dim, hidden_dim, (1, kern_size[1]), padding="same")
        self.gconvs_2 = gnn_model(hidden_dim, heads * hidden_dim, groups)
        self.bns_2 = nn.BatchNorm2d(heads * hidden_dim)

        self.tconvs_3 = nn.Conv2d(hidden_dim, out_dim, (1, kern_size[2]), padding="same")
        self.gconvs_3 = gnn_model(out_dim, heads * out_dim, groups)
        self.bns_3 = nn.BatchNorm2d(heads * out_dim)

        self.linear = nn.Linear(heads * out_dim, num_classes)

        self.dropout_1 = args.dropout_size[0]  # 0.5
        self.dropout_2 = args.dropout_size[1]  # 0.5
        self.dropout_3 = args.dropout_size[2]  # 0.5
        self.activation = nn.ReLU()  # LeakyReLU()  

        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.reset_parameters()

    def reset_parameters(self):
        self.tconvs_1.reset_parameters()
        self.gconvs_1.reset_parameters()
        self.bns_1.reset_parameters()
        self.tconvs_2.reset_parameters()
        self.gconvs_2.reset_parameters()
        self.bns_2.reset_parameters()
        self.tconvs_3.reset_parameters()
        self.gconvs_3.reset_parameters()
        self.bns_3.reset_parameters()
        self.tconvs_tmp_1.reset_parameters()
        self.tconvs_tmp_2.reset_parameters()
        self.tconvs_tmp_3.reset_parameters()
        self.linear.reset_parameters()

    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1

    def pretrain(self, x_in):
        return x_in

    def forward(self, inputs: Tensor):
        # inputs = self.pos_drop(inputs)  # Heartbeat  ArrowHead ...
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            inputs = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            inputs = inputs
        fre_asb = self.asb(inputs)
        B = inputs.size(0)
        _ = inputs.size(1)
        N = inputs.size(-2)
        L = inputs.size(-1)
        inputs = inputs.reshape(B, -1, N, L)
        fre_asb = fre_asb.reshape(B, -1, N, L)
        adj = self.g_constr(inputs.device)
        adj_fre = self.g_constr_fre(inputs.device)

        x_1 = self.tconvs_1(inputs)
        x_1 = self.bns_1(self.activation(x_1))
        x_1 = self.gconvs_1(x_1,adj)

        x_fre_1 = self.tconvs_1(fre_asb)
        x_fre_1 = self.gconvs_1(x_fre_1,adj_fre)

        x_1_out = torch.cat([x_1,x_fre_1],dim=1)
        x_1_out = self.tconvs_tmp_1(x_1_out)

        x_1_out = F.dropout(x_1_out, p=self.dropout_1, training=self.training)

        x_2 = self.tconvs_2(x_1_out)
        x_2 = self.bns_2(self.activation(x_2))
        x_2 = self.gconvs_2(x_2,adj)

        x_b,x_e,x_n,x_l = x_fre_1.size(0),x_fre_1.size(1),x_fre_1.size(2),x_fre_1.size(-1)
        fre_asb = x_fre_1.reshape(x_b*x_e, x_n, x_l)
        fre_asb = self.asb(fre_asb)
        x_fre = fre_asb.reshape(x_b,x_e,x_n,x_l)

        x_fre_2 = self.tconvs_2(x_fre)
        x_fre_2 = self.gconvs_2(x_fre_2,adj_fre)

        x_2 = torch.cat([x_2,x_fre_2],dim=1)
        x_2 = self.tconvs_tmp_2(x_2)
        x_2_out = F.dropout(x_2, p=self.dropout_2, training=self.training)

        x_3 = self.tconvs_3(x_2_out)
        x_3 = self.bns_3(self.activation(x_3))
        x_3 = self.gconvs_3(x_3,adj)

        # x_b,x_e,x_n,x_l = x_fre_2.size(0),x_fre_2.size(1),x_fre_2.size(2),x_fre_2.size(-1)
        # fre_asb = x_fre_2.reshape(x_b*x_e, x_n, x_l)
        # fre_asb = self.asb_1(fre_asb)
        # x_fre = fre_asb.reshape(x_b,x_e,x_n,x_l)
        #
        # x_fre = self.tconvs_3(x_fre)
        # x_fre = self.gconvs_3(x_fre,adj_fre)
        #
        # x_3 = torch.cat([x_3,x_fre],dim=1)
        # x_3 = self.tconvs_tmp_3(x_3)

        x_out = F.dropout(x_3, p=self.dropout_3, training=self.training)

        out = self.global_pool(x_out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = GNNStack()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]

        preds, target = self.model.pretrain(data)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = GNNStack()  # ------
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.parameters(), lr=args.train_lr, weight_decay=1e-4)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs

    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(42)  # To be reproducible
    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)

    return pretrain_checkpoint_callback.best_model_path
def train_model(pretrained_model_path):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="gpu",  # "auto"
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(42)  # 42 To be reproducible
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()

    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint after training
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # model = model_training.load_from_checkpoint("./lightning_logs/Worms_numLayers3_inDim_54_hiddenDim_108_outDim_216_kernSize_[7, 5, 3]_kernSizeMid_[6, 5, 3]_dropoutSize_[0.4, 0.5, 0.5]14_12_51/epoch=287-step=2592.ckpt")

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    acc_test = str(test_result[0]["test_acc"])
    acc_test = float(acc_test[:5])

    acc_val = str(val_result[0]["test_acc"])
    acc_val = float(acc_val[:5])

    f1_test = str(test_result[0]["test_f1"])
    f1_test = float(f1_test[:5])

    f1_val = str(val_result[0]["test_f1"])
    f1_val = float(f1_val[:5])

    acc_result = {"test": acc_test, "val": acc_val}
    f1_result = {"test": f1_test, "val": f1_val}
    # acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    # f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}

    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)

    return model, acc_result, f1_result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='DTFPNet_UCR')
    parser.add_argument('--data_path', type=str, default=r'../dataset/UCR/Worms')
    # parser.add_argument('--data_path', type=str, default=r'../dataset/har')
    # parser.add_argument('--data_path', type=str, default=r'../dataset/UEA/RacketSports')

    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=700)  # 190 200 400 500 600 700 1500 2000
    parser.add_argument('--in_dim', type=int, default=54, help='input dimensions of GNN stacks')  # 64
    parser.add_argument('--hidden_dim', type=int, default=108, help='hidden dimensions of GNN stacks')  # 128
    parser.add_argument('--out_dim', type=int, default=216, help='output dimensions of GNN stacks')  # 256
    parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')  # 3
    parser.add_argument('--groups', type=int, default=2, help='the number of time series groups (num_graphs)')  # 2,3,4,5,6,8
    parser.add_argument('--kern_size', type=str, default=[7,5,3], help='list of time conv kernel size for each layer')
    parser.add_argument('--kern_size_mid', type=str, default=[6,5,3], help='list of time conv kernel size for each layer')
    parser.add_argument('--dropout_size', type=str, default=[0.4,0.5,0.5], help='list of time conv kernel size for each layer')

    parser.add_argument('--GRU_layers', type=int, default=1, help='layers of GRU')  # 1,2,4,12,32

    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)    # 16    FaceDetection_4
    parser.add_argument('--train_lr', type=float, default=5e-4)  # 5e-4  1e-4
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)

    # Model parameters:
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)

    # Pretraining:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False, help='False: without pretraining')

    # DTFPNet parameters:
    parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
    parser.add_argument('--val-batch-size', default=16, type=int, metavar='V',
                        help='validation batch size')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)

    # load from checkpoint   ----------------
    run_description = f"{os.path.basename(args.data_path)}_numLayers{args.num_layers}_inDim_{args.in_dim}_"
    run_description += f"hiddenDim_{args.hidden_dim}_outDim_{args.out_dim}_kernSize_{args.kern_size}_kernSizeMid_{args.kern_size_mid}_dropoutSize_{args.dropout_size}"
    run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_description}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(pretrain_checkpoint_callback)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    print("Dataset loaded ...")

    # Get dataset characteristics ...
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    if args.load_from_pretrained:
        best_model_path = pretrain_model()
    else:
        best_model_path = ''

    model, acc_results, f1_results = train_model(best_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # append result to a text file...
    text_save_dir = "textFiles"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"Ours_{os.path.basename(args.data_path)}_groups_{args.groups}_GRULayers_{args.GRU_layers}_epochs_{args.num_epochs}_train_lr_{args.train_lr}_DCT_GRU" + "  \n")
    f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
    f.write('\n')
    f.write('\n')
    f.close()
