"""
Modèles de prévision charge : entrée (batch, 168, F) → sortie (batch, 24).

Trois variantes : LSTM, Bi-LSTM, LSTM + attention additive (Bahdanau).
Les hyperparamètres par défaut sont des points de départ raisonnables ;
l'entraînement les fixera via ``config.yaml`` (étape suivante).
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Dernière sortie temporelle de la LSTM → tête linéaire vers l'horizon."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))


class BiLSTMForecaster(nn.Module):
    """Bi-LSTM : concat avant/arrière au dernier pas → tête linéaire."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))


class LSTMBahdanauForecaster(nn.Module):
    """
    Encodeur LSTM + contexte d'attention sur les T pas (scores softmax sur le temps).
    Requête = dernière sortie cachée du dernier étage ; alignement additif puis agrégation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.2,
        attn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.w_h = nn.Linear(hidden_size, attn_dim, bias=True)
        self.w_q = nn.Linear(hidden_size, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        y_hat : (B, horizon)
        alpha : (B, T) poids d'attention (pour heatmap / interprétation)
        """
        out, (h_n, _) = self.lstm(x)
        # Dernier étage, état caché final : (B, H)
        query = h_n[-1]
        # out: (B, T, H)
        qh = self.w_q(query).unsqueeze(1)
        energy = torch.tanh(self.w_h(out) + qh)
        scores = self.v(energy).squeeze(-1)
        alpha = torch.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), out).squeeze(1)
        fused = torch.cat([context, query], dim=-1)
        y_hat = self.head(self.dropout(fused))
        return y_hat, alpha


def build_model(
    name: str,
    input_dim: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    horizon: int = 24,
    dropout: float = 0.2,
    attn_dim: int = 64,
) -> nn.Module:
    key = name.strip().lower().replace("-", "_")
    if key in ("lstm",):
        return LSTMForecaster(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout,
        )
    if key in ("bilstm", "bi_lstm"):
        return BiLSTMForecaster(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout,
        )
    if key in ("lstm_attention", "lstm_attn", "attention"):
        return LSTMBahdanauForecaster(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout,
            attn_dim=attn_dim,
        )
    raise ValueError(
        f"Modèle inconnu : {name!r}. Attendu : 'lstm', 'bilstm', 'lstm_attention'."
    )


def _smoke_test() -> None:
    B, T, F, H = 4, 168, 6, 24
    x = torch.randn(B, T, F)
    for name in ("lstm", "bilstm", "lstm_attention"):
        m = build_model(name, input_dim=F, horizon=H)
        if name == "lstm_attention":
            y, alpha = m(x)
            assert y.shape == (B, H)
            assert alpha.shape == (B, T)
        else:
            y = m(x)
            assert y.shape == (B, H)
    print("Smoke test OK : lstm, bilstm, lstm_attention.")


if __name__ == "__main__":
    _smoke_test()
