

import argparse
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_for_sensor import FatigueDataset, PAPER_HYPERPARAMS
from rppg_utils import RPPG_DIM, extract_hrv_from_tensor

warnings.filterwarnings('ignore')


class Config:
    data_root = '.'
    dataset = 'RLDD'  # 'RLDD' | 'NTHU-DDD'

    seq_len = PAPER_HYPERPARAMS['seq_len']
    img_size = PAPER_HYPERPARAMS['img_size']
    fps = PAPER_HYPERPARAMS['fps']
    batch_size = PAPER_HYPERPARAMS['batch_size']
    epochs = PAPER_HYPERPARAMS['epochs']
    lr = PAPER_HYPERPARAMS['lr']
    weight_decay = PAPER_HYPERPARAMS['weight_decay']

    kernel_sizes = [3, 7, 15]  
    num_classes = 2

    lambda_reg = 0.1
    lambda_adv = 0.1
    lambda_mi = 0.05
    lambda_recon = 0.5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiment_mode = 'full'
    use_mtaf = True
    use_pbcaa = True


config = Config()


def parse_batch(batch):
    frames, labels = batch[0], batch[1]
    if len(batch) >= 3:
        scores = batch[2]
        if not torch.is_tensor(scores):
            scores = torch.tensor(scores, dtype=torch.float32)
    else:
        scores = labels.float() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.float32)
    return frames, labels, scores


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.attn = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x)


class SceneAwareModulator(nn.Module):
    """Eq.(2): C = sigma(Linear(GlobalAvgPool(Xenc)))"""

    def __init__(self, in_channels):
        super().__init__()
        self.context_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.context_fc(x.mean(dim=2))


class MultiScaleTemporalAdaptiveFusion(nn.Module):
    """
    多尺度时序自适应融合 MTAF
    Eq.(3): W = Softmax(MLP([Fshort; Fmid; Flong] ⊙ C))
    Eq.(4): Ffused = α·Fshort + β·Fmid + γ·Flong
    """

    def __init__(self, in_channels, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.branches = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for k in self.kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, in_channels, k, padding=k // 2),
                nn.BatchNorm1d(in_channels),
            ))
            self.attentions.append(TemporalAttention(in_channels))

        self.scene_modulator = SceneAwareModulator(in_channels)
        self.weight_mlp = nn.Sequential(
            nn.Linear(in_channels * len(kernel_sizes), in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, len(kernel_sizes)),
        )
        self.fusion_conv = nn.Conv1d(in_channels, in_channels, 1)
        self.fusion_bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        branch_outputs, branch_pooled = [], []
        for branch, attn in zip(self.branches, self.attentions):
            feat = attn(F.relu(branch(x)))
            branch_outputs.append(feat)
            branch_pooled.append(feat.mean(dim=2))

        context = self.scene_modulator(x)
        concat = torch.cat(branch_pooled, dim=1)
        context_exp = context.unsqueeze(1).expand(-1, len(self.kernel_sizes), -1).reshape(concat.shape)
        weights = F.softmax(self.weight_mlp(concat * context_exp), dim=1)

        fused = sum(
            weights[:, i:i + 1].unsqueeze(-1) * out
            for i, out in enumerate(branch_outputs)
        )
        return F.relu(self.fusion_bn(self.fusion_conv(fused))), weights


class PhysiologicalBehavioralCrossModalAlignment(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, rppg_dim=RPPG_DIM):
        super().__init__()
        self.physiology_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, rppg_dim), nn.ReLU(inplace=True),
            nn.Linear(rppg_dim, rppg_dim),
        )
        self.behavior_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, rppg_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(rppg_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(rppg_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.mi_estimator = nn.Sequential(
            nn.Linear(rppg_dim * 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visual_feat):
        rppg_feat = self.physiology_encoder(visual_feat)
        behavior_feat = self.behavior_encoder(visual_feat)
        aligned_feat = (rppg_feat + behavior_feat) / 2
        return rppg_feat, aligned_feat

    def discriminate(self, feat):
        return self.discriminator(feat)

    def reconstruct(self, rppg_feat):
        return self.decoder(rppg_feat)

    def mutual_information(self, rppg_feat, behavior_feat):
        return self.mi_estimator(torch.cat([rppg_feat, behavior_feat], dim=1))


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(1, 2))

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]


class MTFANet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.spatial_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_fc = nn.Linear(2048, 512)

        self.temporal_dim = 512
        self.pos_encoding = PositionalEncoding1D(self.temporal_dim, max_len=cfg.seq_len + 10)
        self.use_mtaf = cfg.use_mtaf
        self.use_pbcaa = cfg.use_pbcaa

        if self.use_mtaf:
            self.mtaf = MultiScaleTemporalAdaptiveFusion(self.temporal_dim, cfg.kernel_sizes)
        else:
            k = cfg.kernel_sizes[0] if cfg.kernel_sizes else 3
            self.single_temporal = nn.Sequential(
                nn.Conv1d(self.temporal_dim, self.temporal_dim, k, padding=k // 2),
                nn.BatchNorm1d(self.temporal_dim), nn.ReLU(inplace=True),
            )

        if self.use_pbcaa:
            self.pbcaa = PhysiologicalBehavioralCrossModalAlignment(self.temporal_dim, rppg_dim=RPPG_DIM)

        self.rppg_dim = RPPG_DIM
        joint_dim = self.temporal_dim + (self.rppg_dim if self.use_pbcaa else 0)
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, cfg.num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(joint_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x, return_aux=False):
        B, _, T, _, _ = x.shape
        spatial_feats = []
        for t in range(T):
            feat = self.spatial_encoder(x[:, :, t])
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(B, -1)
            spatial_feats.append(self.spatial_fc(feat))

        temporal_feat = self.pos_encoding(torch.stack(spatial_feats, dim=1).permute(0, 2, 1))

        if self.use_mtaf:
            fused_feat, mtaf_weights = self.mtaf(temporal_feat)
        else:
            fused_feat = self.single_temporal(temporal_feat)
            mtaf_weights = torch.full((B, 3), 1.0 / 3.0, device=x.device)

        fused_pooled = fused_feat.mean(dim=2)

        if self.use_pbcaa:
            rppg_feat, aligned_feat = self.pbcaa(fused_pooled)
            concat_feat = torch.cat([fused_pooled, aligned_feat], dim=1)
        else:
            rppg_feat = torch.zeros(B, self.rppg_dim, device=x.device)
            aligned_feat = rppg_feat
            concat_feat = fused_pooled

        logits = self.classifier(concat_feat)
        fatigue_score = self.regressor(concat_feat).squeeze(-1)

        if return_aux:
            return logits, fatigue_score, {
                'fused_feat': fused_pooled,
                'rppg_feat': rppg_feat,
                'aligned_feat': aligned_feat,
                'mtaf_weights': mtaf_weights,
            }
        return logits, fatigue_score


class PBCAALoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lambda_reg = cfg.lambda_reg
        self.lambda_adv = cfg.lambda_adv
        self.lambda_mi = cfg.lambda_mi
        self.lambda_recon = cfg.lambda_recon

    def info_nce_loss(self, pbcaa_module, rppg_feat):
        B = rppg_feat.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=rppg_feat.device)
        scores = torch.zeros(B, B, device=rppg_feat.device)
        for i in range(B):
            for j in range(B):
                scores[i, j] = pbcaa_module.mutual_information(
                    rppg_feat[i:i + 1], rppg_feat[j:j + 1]
                ).squeeze()
        return F.cross_entropy(scores, torch.arange(B, device=rppg_feat.device))

    def forward(self, pbcaa_module, aux, hrv_targets=None, fatigue_scores=None, pred_scores=None):
        rppg_feat = aux['rppg_feat']
        aligned_feat = aux['aligned_feat']
        fused_feat = aux['fused_feat']

        loss_adv = (
            F.binary_cross_entropy(pbcaa_module.discriminate(rppg_feat.detach()), torch.zeros(rppg_feat.size(0), 1, device=rppg_feat.device)) +
            F.binary_cross_entropy(pbcaa_module.discriminate(aligned_feat.detach()), torch.ones(rppg_feat.size(0), 1, device=rppg_feat.device))
        ) / 2
        loss_mi = self.info_nce_loss(pbcaa_module, rppg_feat)
        loss_recon = F.mse_loss(rppg_feat, hrv_targets) if hrv_targets is not None else F.mse_loss(
            pbcaa_module.reconstruct(rppg_feat), fused_feat
        )
        loss_reg = torch.tensor(0.0, device=fused_feat.device)
        if fatigue_scores is not None and pred_scores is not None:
            loss_reg = F.mse_loss(pred_scores, fatigue_scores.float())

        total = self.lambda_adv * loss_adv + self.lambda_mi * loss_mi + self.lambda_recon * loss_recon + self.lambda_reg * loss_reg
        return total, {
            'loss_adv': loss_adv.item(), 'loss_mi': loss_mi.item(),
            'loss_recon': loss_recon.item(), 'loss_reg': float(loss_reg.item() if torch.is_tensor(loss_reg) else 0.0),
        }


class Trainer:
    def __init__(self, model, cfg):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(cfg.epochs, 1))
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_pbcaa = PBCAALoss(cfg)
        self.history = defaultdict(list)

    def _extract_hrv_batch(self, frames):
        hrv = [extract_hrv_from_tensor(frames[i], fps=self.cfg.fps) for i in range(frames.size(0))]
        return torch.tensor(np.stack(hrv), dtype=torch.float32, device=frames.device)

    def train_epoch(self, loader):
        self.model.train()
        total_loss, preds, labels = 0.0, [], []

        for batch in tqdm(loader, desc='Training'):
            frames, y, scores = parse_batch(batch)
            frames, y, scores = frames.to(self.cfg.device), y.to(self.cfg.device), scores.to(self.cfg.device)

            self.optimizer.zero_grad()
            logits, pred_scores, aux = self.model(frames, return_aux=True)
            loss = self.criterion_cls(logits, y)

            if getattr(self.model, 'use_pbcaa', True) and hasattr(self.model, 'pbcaa'):
                hrv = self._extract_hrv_batch(frames)
                if hrv.size(-1) != aux['rppg_feat'].size(-1):
                    hrv = hrv[:, :aux['rppg_feat'].size(-1)]
                loss_pbcaa, _ = self.criterion_pbcaa(
                    self.model.pbcaa, aux, hrv_targets=hrv,
                    fatigue_scores=scores, pred_scores=pred_scores,
                )
                loss = loss + loss_pbcaa

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())

        self.scheduler.step()
        n = max(len(loader), 1)
        return total_loss / n, accuracy_score(labels, preds), f1_score(labels, preds, average='weighted')

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        preds, labels, weights = [], [], []

        for batch in tqdm(loader, desc='Evaluating'):
            frames, y, _ = parse_batch(batch)
            frames, y = frames.to(self.cfg.device), y.to(self.cfg.device)
            logits, _, aux = self.model(frames, return_aux=True)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            weights.append(aux['mtaf_weights'].cpu().numpy())

        if weights:
            w = np.mean(np.concatenate(weights, axis=0), axis=0)
            print(f'  MTAF learned mean weights [short, mid, long]: [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]')

        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted'),
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(labels, preds),
        }

    def train(self, train_loader, val_loader):
        for epoch in range(self.cfg.epochs):
            tr_loss, tr_acc, tr_f1 = self.train_epoch(train_loader)
            val_m = self.evaluate(val_loader)
            self.history['train_loss'].append(tr_loss)
            self.history['val_acc'].append(val_m['accuracy'])
            print(f'Epoch {epoch + 1}/{self.cfg.epochs}  train_loss={tr_loss:.4f}  val_acc={val_m["accuracy"]:.4f}  val_f1={val_m["f1"]:.4f}')
        return self.history

class BaselineVGG16LSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=256):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children()))
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.lstm = nn.LSTM(512, lstm_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        B, _, T, _, _ = x.shape
        feats = []
        for t in range(T):
            f = self.pool(self.features(x[:, :, t])).view(B, -1)
            feats.append(self.fc(f))
        out, _ = self.lstm(torch.stack(feats, dim=1))
        return self.classifier(out.mean(dim=1))


class Baseline3DResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_conv = nn.Conv3d(2048, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        B, _, T, _, _ = x.shape
        feats = [self.pool(self.features(x[:, :, t])).squeeze(-1).squeeze(-1) for t in range(T)]
        feats = self.temporal_conv(torch.stack(feats, dim=2).unsqueeze(-1).unsqueeze(-1))
        return self.classifier(feats.squeeze(-1).squeeze(-1).mean(dim=2))


class BaselineViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.proj = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        B, _, T, _, _ = x.shape
        feats = []
        for t in range(T):
            frame = x[:, :, t]
            if frame.shape[-1] != 224:
                frame = F.interpolate(frame, size=(224, 224), mode='bilinear', align_corners=False)
            feats.append(self.vit(frame))
        return self.classifier(self.proj(torch.stack(feats, dim=1).mean(dim=1)))


def build_model(cfg):
    if cfg.experiment_mode == 'baseline_vgg_lstm':
        return BaselineVGG16LSTM(cfg.num_classes)
    if cfg.experiment_mode == 'baseline_3d_resnet':
        return Baseline3DResNet50(cfg.num_classes)
    if cfg.experiment_mode == 'baseline_vit':
        return BaselineViT(cfg.num_classes)
    return MTFANet(cfg)


def build_loaders(cfg):
    train_ds = FatigueDataset(cfg.data_root, cfg.dataset, 'train', cfg.seq_len, cfg.img_size)
    val_ds = FatigueDataset(cfg.data_root, cfg.dataset, 'val', cfg.seq_len, cfg.img_size)
    cfg.num_classes = train_ds.num_classes
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, len(train_ds), len(val_ds)


def run_demo(cfg):
    print('=' * 60)
    print('MTFA-Net — Reviewer Smoke Test (demo only)')
    print('=' * 60)
    print(f'dataset={cfg.dataset}  device={cfg.device}')
    print(f'demo settings: seq_len={cfg.seq_len}  epochs={cfg.epochs}  batch={cfg.batch_size}')
    print('This run is for code sanity check only.')
    print('It does NOT reproduce paper tables or report final benchmark numbers.')
    print('=' * 60)

    train_loader, val_loader, n_train, n_val = build_loaders(cfg)
    print(f'samples: train={n_train}, val={n_val}, classes={cfg.num_classes}')

    model = build_model(cfg)
    trainer = Trainer(model, cfg)
    trainer.train(train_loader, val_loader)
    metrics = trainer.evaluate(val_loader)

    print('\n=== Demo Metrics (sanity check only, not for benchmarking) ===')
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f'  {k}: {v:.4f}')
    print(f'  confusion_matrix:\n{metrics["confusion_matrix"]}')
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='MTFA-Net method code — reviewer smoke test only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data_root', default='.', help='Root with archive/ and/or NTHU-DDD/')
    parser.add_argument('--dataset', default='RLDD', choices=['RLDD', 'NTHU-DDD'])

    args = parser.parse_args()

    cfg = Config()
    cfg.data_root = args.data_root
    cfg.dataset = args.dataset
    cfg.experiment_mode = 'full'
    cfg.seq_len = 16
    cfg.epochs = 2
    cfg.batch_size = 4

    run_demo(cfg)


if __name__ == '__main__':
    main()
