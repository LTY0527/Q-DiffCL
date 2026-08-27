from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from losses import supervised_contrastive_loss
from models.backbones import TCNClassifier


OFFICIAL_COMMITS = {
    "TimesURL": "d3533e45cb28efe8c986f13ce8d80926d0e9254e",
    "MF-CLR": "c40fc8d265947f7a194ac43b8256c2b5d9febe01",
    "REBAR": "74cd46b56262488378f49ebe6ea40ee59ff577dc",
    "TF-C": "96675826e9ef234a9b01cc63d484c66cb0441bc0",
    "TS2Vec": "b0088e14a99706c05451316dc6db8d3da9351163",
    "SoftCLT": "14c638979b129075d7a1111e9f529b9a275ea394",
}
TRACKS = {
    "TimesURL": "TRACK_B_METHOD_NATIVE_REPRESENTATION",
    "MF-CLR": "TRACK_B_METHOD_NATIVE_REPRESENTATION",
    "REBAR": "TRACK_B_METHOD_NATIVE_REPRESENTATION",
    "AutoTCL": "TRACK_A_MECHANISM_ADAPTATION",
    "TF-C": "TRACK_B_METHOD_NATIVE_REPRESENTATION",
    "SoftCLT": "TRACK_A_MECHANISM_ADAPTATION",
    "TS2Vec": "TRACK_B_METHOD_NATIVE_REPRESENTATION",
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def bcl_to_btc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("expected [N,C,L]")
    result = np.ascontiguousarray(values.transpose(0, 2, 1))
    if not np.isfinite(result).all():
        raise ValueError("baseline input contains non-finite values")
    return result


def deterministic_grain_split(channels: int) -> list[int]:
    if channels < 2:
        return [max(1, channels), max(1, channels)]
    boundary = max(1, channels // 2)
    return [boundary, channels]


def verify_external_commit(root: str | Path, method: str) -> str:
    import subprocess
    root = Path(root)
    expected = OFFICIAL_COMMITS[method]
    if not (root / ".git").is_dir():
        raise RuntimeError(f"missing official clone for {method}: {root}")
    observed = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True, encoding="utf-8").strip()
    if observed != expected:
        raise RuntimeError(f"{method} official commit changed: {observed} != {expected}")
    return observed


@contextlib.contextmanager
def isolated_import(root: str | Path, module: str, namespaces: tuple[str, ...]) -> Iterator[Any]:
    """Load official repos whose unnamespaced modules collide with Q-DiffCL.

    Imported objects retain their module globals after the repository modules are
    removed from sys.modules, while Q-DiffCL's original modules are restored.
    """
    saved = {key: value for key, value in list(sys.modules.items())
             if key.split(".", 1)[0] in namespaces}
    for key in saved:
        sys.modules.pop(key, None)
    original_path = list(sys.path)
    try:
        resolved = Path(root).resolve(); sys.path.insert(0, str(resolved))
        # REBAR uses implicit namespace packages. A later regular Q-DiffCL
        # package would otherwise win the import search despite sys.path order.
        for name in namespaces:
            directory = resolved / name
            if directory.is_dir() and not (directory / "__init__.py").exists():
                package = types.ModuleType(name); package.__path__ = [str(directory)]  # type: ignore[attr-defined]
                package.__package__ = name; sys.modules[name] = package
        yield importlib.import_module(module)
    finally:
        for key in list(sys.modules):
            if key.split(".", 1)[0] in namespaces:
                sys.modules.pop(key, None)
        sys.modules.update(saved)
        sys.path[:] = original_path


def _install_sequential_pandarallel_shim() -> None:
    """Numerically equivalent Windows compatibility for MF-CLR's dataframe map."""
    import pandas as pd
    if not hasattr(pd.DataFrame, "parallel_apply"):
        pd.DataFrame.parallel_apply = pd.DataFrame.apply  # type: ignore[attr-defined]
    module = types.ModuleType("pandarallel")
    module.pandarallel = types.SimpleNamespace(initialize=lambda **_: None)
    sys.modules.setdefault("pandarallel", module)


def _install_tensorboard_shim() -> None:
    if "torch.utils.tensorboard" in sys.modules:
        return
    module = types.ModuleType("torch.utils.tensorboard")
    class SummaryWriter:
        def __init__(self, *args: Any, **kwargs: Any): pass
        def add_scalar(self, *args: Any, **kwargs: Any) -> None: pass
        def close(self) -> None: pass
    module.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = module


class AutoTCLGate(nn.Module):
    """Independent, shared-TCN adaptation of AutoTCL's learned multiplicative gate."""

    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(channels, hidden, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(hidden, channels, 1))
        self.scale = nn.Sequential(nn.Conv1d(channels, hidden, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, values: torch.Tensor, hard: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(self.features(values))
        if self.training:
            uniform = torch.rand_like(probability).clamp_(1e-6, 1 - 1e-6)
            # Formal runs can drive sigmoid outputs to exact zero or one.
            # Clamping only the sampling logit keeps the hard-gate semantics
            # while avoiding an infinite derivative at the probability bounds.
            sampling_probability = probability.clamp(1e-6, 1 - 1e-6)
            soft = torch.sigmoid((torch.logit(sampling_probability) + torch.logit(uniform)) / .5)
        else:
            soft = probability
        if hard:
            binary = (soft >= .5).to(soft.dtype)
            gate = soft + (binary - soft).detach()
        else:
            gate = soft
        changed = values * (gate * self.scale(values))
        return changed, probability


class AutoTCLAdaptation(nn.Module):
    def __init__(self, channels: int, classes: int, model: dict[str, Any]):
        super().__init__()
        self.backbone = TCNClassifier(channels, int(model["hidden_channels"]),
                                      int(model["projection_dim"]), classes, int(model["levels"]))
        self.gate = AutoTCLGate(channels)

    def encode_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return self.backbone(values, projection=False, classification=False)["embedding"]

    def encode(self, values: np.ndarray, batch_size: int, device: str) -> np.ndarray:
        self.eval(); rows = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                x = torch.from_numpy(values[start:start + batch_size]).float().to(device)
                rows.append(self.encode_tensor(x).cpu().numpy())
        return np.concatenate(rows)


class TFCRepresentation(nn.Module):
    """TF-C dual time/frequency Transformer with multivariate mean pooling."""
    def __init__(self, length: int = 64, dimension: int = 64):
        super().__init__()
        time_layer = nn.TransformerEncoderLayer(length, nhead=2, dim_feedforward=2 * length, batch_first=True)
        frequency_layer = nn.TransformerEncoderLayer(length, nhead=2, dim_feedforward=2 * length, batch_first=True)
        self.time_encoder = nn.TransformerEncoder(time_layer, 2)
        self.frequency_encoder = nn.TransformerEncoder(frequency_layer, 2)
        self.time_projector = nn.Sequential(nn.Linear(length, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, dimension))
        self.frequency_projector = nn.Sequential(nn.Linear(length, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, dimension))

    def forward(self, time_values: torch.Tensor, frequency_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h_time = self.time_encoder(time_values).mean(1); h_frequency = self.frequency_encoder(frequency_values).mean(1)
        return h_time, self.time_projector(h_time), h_frequency, self.frequency_projector(h_frequency)

    def encode(self, values: np.ndarray, batch_size: int, device: str) -> np.ndarray:
        self.eval(); rows=[]
        with torch.no_grad():
            for start in range(0,len(values),batch_size):
                x=torch.from_numpy(values[start:start+batch_size]).float().to(device); f=torch.abs(torch.fft.fft(x,dim=-1))
                _,zt,_,zf=self(x,f); rows.append(torch.cat([zt,zf],1).cpu().numpy())
        return np.concatenate(rows).astype(np.float32)


class SoftCLTAdaptation(nn.Module):
    """Independent shared-TCN adaptation of SoftCLT's soft instance targets."""
    def __init__(self, channels: int, classes: int, model: dict[str, Any]):
        super().__init__(); self.backbone=TCNClassifier(channels,int(model["hidden_channels"]),int(model["projection_dim"]),classes,int(model["levels"]))

    def encode(self, values: np.ndarray, batch_size: int, device: str) -> np.ndarray:
        self.eval(); rows=[]
        with torch.no_grad():
            for start in range(0,len(values),batch_size):
                x=torch.from_numpy(values[start:start+batch_size]).float().to(device); rows.append(self.backbone(x,projection=False,classification=False)["embedding"].cpu().numpy())
        return np.concatenate(rows).astype(np.float32)


def _nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    count=len(first); z=F.normalize(torch.cat([first,second]),dim=1); logits=z@z.T/temperature
    logits.fill_diagonal_(-torch.inf); targets=torch.cat([torch.arange(count,2*count,device=z.device),torch.arange(count,device=z.device)])
    return F.cross_entropy(logits,targets)


def fit_tfc(model: TFCRepresentation, train_x: np.ndarray, val_x: np.ndarray, epochs: int, batch_size: int,
            learning_rate: float, seed: int, device: str, max_batches: int | None=None) -> list[dict[str,float]]:
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate); rng=np.random.default_rng(seed); history=[]; best=None; best_loss=float("inf")
    def epoch(values: np.ndarray, training: bool) -> float:
        model.train(training); losses=[]; order=rng.permutation(len(values)) if training else np.arange(len(values))
        context=torch.enable_grad() if training else torch.no_grad()
        with context:
            for number,start in enumerate(range(0,len(order),batch_size)):
                if max_batches is not None and number>=max_batches: break
                idx=order[start:start+batch_size]
                if len(idx)<2: continue
                x=torch.from_numpy(values[idx]).float().to(device); xt=x+.02*torch.randn_like(x); f=torch.abs(torch.fft.fft(x,dim=-1)); fa=f.clone()
                mask=torch.rand_like(fa)<.1; fa=torch.where(mask,torch.zeros_like(fa),fa); ht,zt,hf,zf=model(x,f); hta,zta,hfa,zfa=model(xt,fa)
                loss=.2*(_nt_xent(ht,hta,.2)+_nt_xent(hf,hfa,.2))+_nt_xent(zt,zf,.2)
                if training: optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
        return float(np.mean(losses))
    for index in range(epochs):
        loss=epoch(train_x,True); val=epoch(val_x,False); history.append({"epoch":index,"loss":loss,"validation_loss":val})
        if val<best_loss: best_loss,best=val,copy.deepcopy(model.state_dict())
    if best is not None:model.load_state_dict(best)
    return history


def fit_softclt(model: SoftCLTAdaptation, train_x: np.ndarray, val_x: np.ndarray, epochs: int,batch_size:int,
                learning_rate:float,temperature:float,seed:int,device:str,max_batches:int|None=None)->list[dict[str,float]]:
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate); rng=np.random.default_rng(seed); history=[]; best=None; best_loss=float("inf")
    def epoch(values:np.ndarray,training:bool)->float:
        model.train(training); losses=[]; order=rng.permutation(len(values)) if training else np.arange(len(values)); context=torch.enable_grad() if training else torch.no_grad()
        with context:
            for number,start in enumerate(range(0,len(order),batch_size)):
                if max_batches is not None and number>=max_batches:break
                idx=order[start:start+batch_size]
                if len(idx)<2:continue
                x=torch.from_numpy(values[idx]).float().to(device); x2=x+.02*torch.randn_like(x); z1=F.normalize(model.backbone(x)["projection"],dim=1); z2=F.normalize(model.backbone(x2)["projection"],dim=1)
                raw=F.normalize(x.flatten(1),dim=1); distance=torch.cdist(raw,raw); scale=torch.median(distance.detach()).clamp_min(1e-6); weights=torch.softmax(-distance/scale,dim=1); logits=z1@z2.T/temperature; loss=-(weights*F.log_softmax(logits,dim=1)).sum(1).mean()
                if training:optimizer.zero_grad();loss.backward();optimizer.step()
                losses.append(float(loss.detach()))
        return float(np.mean(losses))
    for index in range(epochs):
        loss=epoch(train_x,True);val=epoch(val_x,False);history.append({"epoch":index,"loss":loss,"validation_loss":val})
        if val<best_loss:best_loss,best=val,copy.deepcopy(model.state_dict())
    if best is not None:model.load_state_dict(best)
    return history


def fit_autotcl(model: AutoTCLAdaptation, train_x: np.ndarray, train_y: np.ndarray,
                val_x: np.ndarray, val_y: np.ndarray, epochs: int, batch_size: int,
                learning_rate: float, temperature: float, seed: int, device: str,
                max_batches: int | None = None) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed); history = []; best = None; best_loss = float("inf")
    for epoch in range(epochs):
        model.train(); losses = []
        order = rng.permutation(len(train_x))
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            if max_batches is not None and batch_index >= max_batches:
                break
            indices = order[start:start + batch_size]
            x = torch.from_numpy(train_x[indices]).float().to(device)
            y = torch.from_numpy(train_y[indices]).long().to(device)
            if len(torch.unique(y)) < 2:
                continue
            optimizer.zero_grad(); changed, probability = model.gate(x)
            if not torch.isfinite(probability).all():
                raise FloatingPointError("non-finite AutoTCL gate probability")
            if not torch.isfinite(changed).all():
                raise FloatingPointError("non-finite AutoTCL changed view")
            clean_projection = model.backbone(x)["projection"]
            changed_projection = model.backbone(changed)["projection"]
            if not torch.isfinite(clean_projection).all():
                raise FloatingPointError("non-finite AutoTCL clean projection")
            if not torch.isfinite(changed_projection).all():
                raise FloatingPointError("non-finite AutoTCL changed projection")
            projections = torch.cat([clean_projection, changed_projection])
            labels = torch.cat([y, y])
            contrast = supervised_contrastive_loss(projections, labels, temperature)
            regular = .001 * torch.mean(torch.abs(probability[:, :, 1:] - probability[:, :, :-1]))
            loss = contrast + regular; loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise FloatingPointError("non-finite AutoTCL gradient")
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); validation = []
        with torch.no_grad():
            for start in range(0, min(len(val_x), batch_size * (max_batches or 10**9)), batch_size):
                x = torch.from_numpy(val_x[start:start + batch_size]).float().to(device)
                y = torch.from_numpy(val_y[start:start + batch_size]).long().to(device)
                changed, _ = model.gate(x)
                z = torch.cat([model.backbone(x)["projection"], model.backbone(changed)["projection"]])
                validation.append(float(supervised_contrastive_loss(z, torch.cat([y, y]), temperature)))
        val_loss = float(np.mean(validation)); record = {"epoch": epoch, "loss": float(np.mean(losses)),
                                                          "validation_loss": val_loss}
        history.append(record)
        if val_loss < best_loss:
            best_loss, best = val_loss, copy.deepcopy(model.state_dict())
    if best is not None:
        model.load_state_dict(best)
    return history


class NativeRecentModel:
    def __init__(self, method: str, channels: int, classes: int, config: dict[str, Any],
                 device: str, run_dir: Path):
        self.method, self.channels, self.classes = method, channels, classes
        self.config, self.device, self.run_dir = config, device, run_dir
        self.model: Any = None
        self.audit: dict[str, Any] = {"method": method, "track": TRACKS[method]}

    def _load_timesurl(self) -> Any:
        root = Path(self.config["external_roots"]["TimesURL"]); self.audit["official_commit"] = verify_external_commit(root, "TimesURL")
        args = Namespace(batch_size=int(self.config["native"]["TimesURL"]["batch_size"]), segment_num=3,
                         mask_ratio_per_seg=.05, lmd=.01)
        with isolated_import(root / "src", "timesurl", ("timesurl", "models", "utils", "lib", "collator")) as module:
            return module.TimesURL(input_dims=self.channels, output_dims=int(self.config["representation_dim"]),
                                   hidden_dims=64, depth=10, device=self.device,
                                   lr=float(self.config["learning_rate"]), batch_size=args.batch_size,
                                   max_train_length=None, args=args)

    def _load_mfclr(self) -> Any:
        root = Path(self.config["external_roots"]["MF-CLR"]); self.audit["official_commit"] = verify_external_commit(root, "MF-CLR")
        _install_sequential_pandarallel_shim()
        split = deterministic_grain_split(self.channels); self.audit["grain_split"] = split
        with isolated_import(root, "MFCLR", ("MFCLR", "encoder", "lossfunc", "util_func", "data_augmentation")) as module:
            return module.MF_CLR(input_dims=split[0], grain_split=split, total_dim=self.channels,
                                 output_dims=int(self.config["representation_dim"]), hidden_dims=256, depth=7,
                                 device=self.device, lr=float(self.config["learning_rate"]),
                                 batch_size=int(self.config["native"]["MF-CLR"]["batch_size"]), projection=True,
                                 da="proposed")

    def _load_rebar(self, train: np.ndarray, validation: np.ndarray, epochs: int) -> Any:
        root = Path(self.config["external_roots"]["REBAR"]); self.audit["official_commit"] = verify_external_commit(root, "REBAR")
        _install_tensorboard_shim()
        with isolated_import(root, "models.REBAR.REBAR_SSLModel", ("models", "experiments", "utils")) as module:
            config_module = importlib.import_module("experiments.configs.rebar_expconfigs")
            cross_module = importlib.import_module("models.REBAR.REBAR_CrossAttn.REBAR_CrossAttn")
            # Official num_workers=torch.get_num_threads() cannot pickle the
            # isolated namespace modules under Windows spawn. Worker count is
            # an I/O choice, so a zero-worker compatibility adapter is numeric-neutral.
            def ssl_loader(instance: Any, data: np.ndarray, train: bool) -> Any:
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
                return torch.utils.data.DataLoader(dataset, batch_size=instance.batch_size, shuffle=train, num_workers=0)
            module.REBAR.setup_dataloader = ssl_loader
            def cross_loader(instance: Any, data: np.ndarray, mask_extended: int | None = None,
                             mask_transient_perc: float | None = None, train: bool = True) -> Any:
                dataset = cross_module.rebarcrossattn_maskdataset(waveforms=data, subseq_size=instance.subseq_size,
                                                                  mask_extended=mask_extended,
                                                                  mask_transient_perc=mask_transient_perc)
                return torch.utils.data.DataLoader(dataset, batch_size=instance.batch_size, shuffle=train, num_workers=0)
            cross_module.REBAR_CrossAttn_Trainer.setup_dataloader_rebarcrossattn = cross_loader
            self.audit["compatibility_patch"] = "no-op tensorboard and Windows DataLoader num_workers=0"
            cross = cross_module.REBAR_CrossAttn_Config(double_receptivefield=1, mask_extended=3,
                                                        rebarcrossattn_epochs=max(1, epochs // 4),
                                                        rebarcrossattn_batch_size=int(self.config["native"]["REBAR"]["batch_size"]),
                                                        rebarcrossattn_save_epochfreq=max(1, epochs))
            cfg = config_module.REBAR_ExpConfig(rebarcrossattn_config=cross, candidateset_size=5, tau=.1, alpha=.5,
                                                data_name="qdiffcl_posthoc", subseq_size=16, epochs=epochs,
                                                lr=float(self.config["learning_rate"]),
                                                batch_size=int(self.config["native"]["REBAR"]["batch_size"]),
                                                save_epochfreq=max(1, epochs), seed=0)
            cfg.set_device(self.device); cfg.set_inputdims(self.channels); cfg.set_rundir(str(self.run_dir.resolve()))
            return module.REBAR(cfg, train_data=train, val_data=validation)

    def _load_ts2vec(self) -> Any:
        root=Path(self.config["external_roots"]["TS2Vec"]);self.audit["official_commit"]=verify_external_commit(root,"TS2Vec")
        with isolated_import(root,"ts2vec",("ts2vec","models","utils")) as module:
            return module.TS2Vec(input_dims=self.channels,output_dims=int(self.config["representation_dim"]),hidden_dims=64,depth=10,device=self.device,lr=float(self.config["learning_rate"]),batch_size=int(self.config["native"]["TS2Vec"]["batch_size"]),max_train_length=None)

    @staticmethod
    def _timesurl_bundle(values: np.ndarray) -> dict[str, np.ndarray]:
        time = np.broadcast_to(np.linspace(0, 1, values.shape[1], dtype=np.float32)[None, :, None],
                               (len(values), values.shape[1], 1))
        return {"x": np.concatenate([values, time], axis=-1), "mask": np.ones_like(values, dtype=np.float32)}

    def fit(self, train_bcl: np.ndarray, validation_bcl: np.ndarray, epochs: int) -> list[Any]:
        train, validation = bcl_to_btc(train_bcl), bcl_to_btc(validation_bcl)
        if self.method == "TimesURL":
            self.model = self._load_timesurl()
            return self.model.fit(self._timesurl_bundle(train), n_epochs=epochs, verbose=True, is_scheduler=False)
        if self.method == "MF-CLR":
            self.model = self._load_mfclr(); return self.model.fit(train, n_epochs=epochs, verbose=True)
        if self.method == "REBAR":
            self.model = self._load_rebar(train, validation, epochs); self.model.fit(); self.model.load("best"); return []
        if self.method == "TS2Vec":
            self.model=self._load_ts2vec();return self.model.fit(train,n_epochs=epochs,verbose=True)
        raise ValueError(self.method)

    def encode(self, values_bcl: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        values = bcl_to_btc(values_bcl)
        if self.method == "TimesURL":
            output = self.model.encode(self._timesurl_bundle(values), encoding_window="full_series", batch_size=batch_size)
        elif self.method == "MF-CLR":
            # Official full_series encode pools the fine branch but concatenates
            # an unpooled coarse branch (length 1 vs T). Pooling both is the
            # shape-correct implementation of its documented full-series view.
            split = self.model.grain_split_list; rows = []; batch = int(batch_size or self.model.batch_size)
            for net in self.model.net_list: net.eval()
            with torch.no_grad():
                for start in range(0, len(values), batch):
                    x = torch.from_numpy(values[start:start + batch]).float().to(self.device)
                    fine = self.model.net_list[0](x[:, :, :split[0]])
                    coarse = self.model.ph_list[0](x[:, :, split[0]:split[1]])
                    fine = torch.amax(fine, dim=1); coarse = torch.amax(coarse, dim=1)
                    rows.append(torch.cat([fine, coarse], dim=1).cpu().numpy())
            output = np.concatenate(rows)
            self.audit["compatibility_patch"] = "pool both fine and coarse branches for documented full_series encoding"
        elif self.method == "REBAR":
            output = self.model.encode(values)
        elif self.method == "TS2Vec":
            output=self.model.encode(values,encoding_window="full_series",batch_size=batch_size)
        else:
            raise ValueError(self.method)
        output = np.asarray(output, dtype=np.float32).reshape(len(values), -1)
        if not np.isfinite(output).all() or len(output) != len(values):
            raise RuntimeError(f"invalid {self.method} representation")
        return output

    def state_dict(self) -> dict[str, Any]:
        if self.method == "TimesURL": return {"net": self.model.net.state_dict()}
        if self.method == "MF-CLR":
            return {"nets": [net.state_dict() for net in self.model.net_list],
                    "projectors": [net.state_dict() for net in self.model.ph_list]}
        if self.method == "REBAR":
            return {"encoder": self.model.encoder.state_dict(), "cross_attention": self.model.rebar_crossattn_trainer.rebarcrossattn_model.state_dict()}
        if self.method == "TS2Vec": return {"net":self.model.net.state_dict()}
        raise ValueError(self.method)

    def load_state_dict(self, state: dict[str, Any], train_bcl: np.ndarray | None = None,
                        validation_bcl: np.ndarray | None = None) -> None:
        if self.method == "TimesURL": self.model = self._load_timesurl(); self.model.net.load_state_dict(state["net"])
        elif self.method == "MF-CLR":
            self.model = self._load_mfclr()
            for net, current in zip(self.model.net_list, state["nets"]): net.load_state_dict(current)
            for net, current in zip(self.model.ph_list, state["projectors"]): net.load_state_dict(current)
        elif self.method == "REBAR":
            if train_bcl is None or validation_bcl is None: raise ValueError("REBAR reconstruction requires train/validation shapes")
            # Construction trains cross-attention, so formal resume uses its official on-disk checkpoint.
            self.model = self._load_rebar(bcl_to_btc(train_bcl), bcl_to_btc(validation_bcl), 0)
            self.model.encoder.load_state_dict(state["encoder"])
            self.model.rebar_crossattn_trainer.rebarcrossattn_model.load_state_dict(state["cross_attention"])
        elif self.method == "TS2Vec":self.model=self._load_ts2vec();self.model.net.load_state_dict(state["net"])
        else: raise ValueError(self.method)


class FrozenLinearProbe(nn.Module):
    def __init__(self, dimension: int, classes: int):
        super().__init__(); self.linear = nn.Linear(dimension, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


def fit_linear_probe(train_z: np.ndarray, train_y: np.ndarray, val_z: np.ndarray, val_y: np.ndarray,
                     classes: int, epochs: int, batch_size: int, learning_rate: float,
                     seed: int, device: str) -> tuple[FrozenLinearProbe, list[dict[str, float]]]:
    torch.manual_seed(seed); model = FrozenLinearProbe(train_z.shape[1], classes).to(device)
    counts = np.bincount(train_y, minlength=classes); weights = np.sqrt(len(train_y) / np.maximum(counts, 1))
    weights /= weights.mean(); criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate); rng = np.random.default_rng(seed)
    history = []; best = None; best_rank = None
    from sklearn.metrics import f1_score
    for epoch in range(epochs):
        model.train(); losses = []; order = rng.permutation(len(train_z))
        for start in range(0, len(train_z), batch_size):
            idx = order[start:start + batch_size]
            x = torch.from_numpy(train_z[idx]).to(device); y = torch.from_numpy(train_y[idx]).long().to(device)
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        probability = probe_probabilities(model, val_z, batch_size, device); prediction = probability.argmax(1)
        score = float(f1_score(val_y, prediction, average="macro", zero_division=0))
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_macro_f1": score}; history.append(record)
        rank = (score, -float(np.mean(prediction[val_y == 0] != 0)) if np.any(val_y == 0) else 0.0)
        if best_rank is None or rank > best_rank: best_rank, best = rank, copy.deepcopy(model.state_dict())
    model.load_state_dict(best); model.eval(); return model, history


def probe_probabilities(model: FrozenLinearProbe, representations: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    rows = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(representations), batch_size):
            x = torch.from_numpy(representations[start:start + batch_size]).float().to(device)
            rows.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(rows)
