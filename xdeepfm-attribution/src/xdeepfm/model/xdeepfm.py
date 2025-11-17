"""PyTorch implementation of xDeepFM."""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn

from ..data.metadata import PreprocessMetadata


def _get_activation(name: str | None) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "identity" or name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class CompressedInteractionNetwork(nn.Module):
    """Faithful CIN implementation that keeps per-dimension signals before the final projection."""

    def __init__(
        self,
        field_dim: int,
        layer_sizes: Sequence[int],
        embedding_dim: int,
        activation: str = "identity",
        *,
        direct_connect: bool = True,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_sizes = [int(size) for size in layer_sizes if int(size) > 0]
        self.embedding_dim = int(embedding_dim)
        self.field_dim = int(field_dim)
        self.activation = _get_activation(activation)
        self.direct_connect = bool(direct_connect)
        self.conv_layers = nn.ModuleList()
        self._split_flags: List[bool] = []
        prev_field_count = self.field_dim
        accumulated_dim = 0
        for idx, size in enumerate(self.layer_sizes):
            in_channels = prev_field_count * self.field_dim
            conv = nn.Conv1d(in_channels=in_channels, out_channels=size, kernel_size=1, bias=True)
            self.conv_layers.append(conv)
            is_last = idx == len(self.layer_sizes) - 1
            if not self.direct_connect and not is_last:
                if size % 2 != 0:
                    raise ValueError("CIN layer size must be even when direct_connect is False.")
                next_hidden = size // 2
                self._split_flags.append(True)
                accumulated_dim += next_hidden
                prev_field_count = next_hidden
            else:
                self._split_flags.append(False)
                accumulated_dim += size
                prev_field_count = size
        self.final_dim = accumulated_dim
        self.output_dim = int(output_dim) if output_dim is not None else self.final_dim
        if self.output_dim <= 0:
            raise ValueError("CIN output dimension must be positive.")
        self.output_proj = nn.Linear(self.final_dim, self.output_dim) if self.output_dim != self.final_dim else nn.Identity()

    def _pairwise_interactions(self, xk: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        # xk: (batch, prev_field, emb_dim), x0: (batch, field_dim, emb_dim)
        outer = torch.einsum("bhd,bmd->bhmd", xk, x0)  # (batch, prev_field, field_dim, emb_dim)
        batch, prev_field, field_dim, emb_dim = outer.shape
        return outer.reshape(batch, prev_field * field_dim, emb_dim)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        if not self.layer_sizes:
            return torch.zeros(x0.size(0), 0, device=x0.device)
        xk = x0
        outputs = []
        for idx, (conv, split_flag) in enumerate(zip(self.conv_layers, self._split_flags)):
            interactions = self._pairwise_interactions(xk, x0)  # (batch, prev_field * field_dim, emb_dim)
            conv_out = conv(interactions)  # (batch, layer_size, emb_dim)
            conv_out = self.activation(conv_out)
            is_last = idx == len(self.layer_sizes) - 1
            if split_flag:
                split_size = conv_out.size(1) // 2
                next_hidden, direct_part = torch.split(conv_out, [split_size, split_size], dim=1)
                xk = next_hidden
            else:
                direct_part = conv_out
                if not is_last:
                    xk = conv_out
            outputs.append(direct_part)
        stacked = torch.cat(outputs, dim=1)  # (batch, final_dim, emb_dim)
        pooled = stacked.sum(dim=2)  # sum over embedding dimension
        return self.output_proj(pooled)


class DeepComponent(nn.Module):
    def __init__(self, input_dim: int, hidden_units: Sequence[int], activation: str, dropout: float, batch_norm: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        act = activation
        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(_get_activation(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = units
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim if hidden_units else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.net:
            return torch.zeros(x.size(0), 0, device=x.device)
        return self.net(x)


class XDeepFM(nn.Module):
    def __init__(
        self,
        metadata: PreprocessMetadata,
        model_cfg: Dict,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.model_cfg = model_cfg
        embedding_dim = int(model_cfg.get("embedding_dim", 10))
        self.categorical_fields = metadata.categorical_fields
        self.numerical_fields = metadata.numerical_fields
        self.num_categorical = len(self.categorical_fields)
        self.num_numerical = len(self.numerical_fields)
        vocab_sizes = [metadata.vocab_sizes[field] for field in self.categorical_fields]

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim)
            for size in vocab_sizes
        ])
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=1)
            for size in vocab_sizes
        ])
        self.linear_dense = nn.Linear(self.num_numerical, 1) if self.num_numerical else None

        fm_cfg = model_cfg.get("fm", {})
        self.use_fm = bool(fm_cfg.get("use_fm", True))
        self.use_first_order_dense = bool(fm_cfg.get("use_first_order_dense", True))
        self.use_first_order_categorical = bool(fm_cfg.get("use_first_order_categorical", True))

        dnn_cfg = model_cfg.get("dnn", {})
        dnn_input_dim = self.num_numerical + self.num_categorical * embedding_dim
        self.dnn = None
        if dnn_cfg.get("use_dnn", True) and dnn_cfg.get("hidden_units"):
            self.dnn = DeepComponent(
                input_dim=dnn_input_dim,
                hidden_units=dnn_cfg.get("hidden_units", []),
                activation=dnn_cfg.get("activation", "relu"),
                dropout=float(dnn_cfg.get("dropout", 0.0)),
                batch_norm=bool(dnn_cfg.get("batch_norm", False)),
            )
        self.dnn_out_dim = self.dnn.output_dim if self.dnn else 0

        cin_cfg = model_cfg.get("cin", {})
        self.cin = None
        self.cin_out_dim = 0
        if cin_cfg.get("use_cin", False):
            layer_sizes = cin_cfg.get("layer_sizes", [])
            self.cin = CompressedInteractionNetwork(
                field_dim=self.num_categorical,
                layer_sizes=layer_sizes,
                embedding_dim=embedding_dim,
                activation=cin_cfg.get("activation", "identity"),
                direct_connect=bool(cin_cfg.get("direct_connect", True)),
                output_dim=cin_cfg.get("output_dim"),
            )
            self.cin_out_dim = self.cin.output_dim

        combined_dim = self.cin_out_dim + self.dnn_out_dim
        fm_dim = 1 if self.use_fm else 0
        combined_dim += fm_dim

        output_cfg = model_cfg.get("output_layer", {}).get("combine", {})
        hidden_units = output_cfg.get("hidden_units", [])
        activation = output_cfg.get("activation", "none")
        layers: List[nn.Module] = []
        prev_dim = combined_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(_get_activation(activation))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, 1))
        self.output_layer = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        categorical = batch["categorical"]
        numerical = batch["numerical"] if self.num_numerical else None
        emb_list = [emb(categorical[:, idx]) for idx, emb in enumerate(self.embeddings)]
        embed_stack = torch.stack(emb_list, dim=1)  # (batch, num_fields, emb_dim)

        fm_out = None
        if self.use_fm:
            first_order_terms: List[torch.Tensor] = []
            if self.use_first_order_categorical:
                first_order_terms.extend(layer(categorical[:, idx]) for idx, layer in enumerate(self.linear_embeddings))
            if self.use_first_order_dense and self.linear_dense is not None and numerical is not None:
                first_order_terms.append(self.linear_dense(numerical))
            if first_order_terms:
                stacked_terms = torch.stack(first_order_terms, dim=0)
                first_order = stacked_terms.sum(dim=0)
            else:
                first_order = torch.zeros(categorical.size(0), 1, device=categorical.device)
            summed = embed_stack.sum(dim=1)
            summed_square = summed.pow(2).sum(dim=1, keepdim=True)
            square_sum = embed_stack.pow(2).sum(dim=1).sum(dim=1, keepdim=True)
            second_order = 0.5 * (summed_square - square_sum)
            fm_out = first_order + second_order

        dnn_out = None
        if self.dnn is not None:
            flatten_embed = embed_stack.view(categorical.size(0), -1)
            dnn_input_parts = [flatten_embed]
            if numerical is not None:
                dnn_input_parts.insert(0, numerical)
            dnn_input = torch.cat(dnn_input_parts, dim=1)
            dnn_out = self.dnn(dnn_input)

        cin_out = None
        if self.cin is not None:
            cin_out = self.cin(embed_stack)

        parts: List[torch.Tensor] = []
        if fm_out is not None:
            parts.append(fm_out)
        if dnn_out is not None and dnn_out.numel():
            parts.append(dnn_out)
        if cin_out is not None and cin_out.numel():
            parts.append(cin_out)
        if not parts:
            raise RuntimeError("No components enabled in XDeepFM; enable FM, DNN, or CIN.")
        combined = torch.cat(parts, dim=1)
        logits = self.output_layer(combined)
        return logits.squeeze(-1)


def build_model(metadata: PreprocessMetadata, model_cfg: Dict) -> XDeepFM:
    return XDeepFM(metadata, model_cfg)
