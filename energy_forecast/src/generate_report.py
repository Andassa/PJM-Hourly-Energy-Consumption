"""
Génère ``energy_forecast/results/report.html`` (résumé, hyperparamètres, métriques, figures).

Usage (racine du dépôt) :
    python energy_forecast/src/generate_report.py --config energy_forecast/config.yaml
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pick_full_test_metrics(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    if "n_points" not in df.columns:
        return df
    full = df[df["n_points"] >= 300_000].copy()
    if full.empty:
        full = df.drop_duplicates(subset=["model"], keep="last")
    return full.sort_values("mae_mw")


def rel_plot(name: str) -> str:
    return f"plots/{name}"


def img_block(src: str, caption: str) -> str:
    src_esc = html.escape(src)
    cap_esc = html.escape(caption)
    return f"""
    <figure class="fig">
      <img src="{src_esc}" alt="{cap_esc}" loading="lazy" />
      <figcaption>{cap_esc}</figcaption>
    </figure>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère report.html dans results/")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg_path = (args.config or default_config_path()).resolve()
    root = cfg_path.parent
    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    results_dir = root / "results"
    metrics_path = results_dir / "metrics.csv"
    meta_path = root / paths["sequence_meta"]
    plots_dir = results_dir / "plots"
    out_path = results_dir / "report.html"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics_df = pick_full_test_metrics(metrics_path)

    best_row = metrics_df.loc[metrics_df["mae_mw"].idxmin()] if len(metrics_df) else None
    best_name = str(best_row["model"]) if best_row is not None else "—"

    metrics_rows_html = ""
    for _, r in metrics_df.iterrows():
        highlight = ' class="best"' if best_row is not None and r.name == best_row.name else ""
        metrics_rows_html += (
            f"<tr{highlight}>"
            f"<td>{html.escape(str(r['model']))}</td>"
            f"<td>{r['mae_mw']:.1f}</td>"
            f"<td>{r['rmse_mw']:.1f}</td>"
            f"<td>{r['mape_pct']:.2f}</td>"
            f"</tr>\n"
        )

    model_labels = {
        "lstm": "LSTM",
        "bilstm": "Bi-LSTM",
        "lstm_attention": "LSTM + attention (Bahdanau)",
    }

    eval_figures = ""
    for key, label in model_labels.items():
        scatter = plots_dir / f"eval_best_{key}_scatter.png"
        window = plots_dir / f"eval_best_{key}_window_1280.png"
        if scatter.is_file():
            eval_figures += img_block(rel_plot(scatter.name), f"{label} — nuage observé vs prédit (test, MW)")
        if window.is_file():
            eval_figures += img_block(rel_plot(window.name), f"{label} — exemple d’horizon 24 h (fenêtre test)")

    loss_all = plots_dir / "loss_all_models.png"
    loss_section = ""
    if loss_all.is_file():
        loss_section = img_block(rel_plot(loss_all.name), "Courbes MSE train / validation (données normalisées)")
    for key, label in model_labels.items():
        p = plots_dir / f"loss_{key}.png"
        if p.is_file():
            loss_section += img_block(rel_plot(p.name), f"{label} — détail des pertes par époque")

    dm_path = results_dir / "dm_results.csv"
    dm_section = ""
    if dm_path.is_file():
        dm_df = pd.read_csv(dm_path)
        dm_rows = ""
        for _, r in dm_df.iterrows():
            stat = r["dm_stat"]
            pval = r["p_value"]
            stat_s = f"{stat:.4f}" if pd.notna(stat) else "—"
            pval_s = f"{pval:.4f}" if pd.notna(pval) else "—"
            dm_rows += (
                f"<tr><td>{html.escape(str(r['model_a']))}</td>"
                f"<td>{html.escape(str(r['model_b']))}</td>"
                f"<td>{r.get('loss', '')}</td>"
                f"<td>{stat_s}</td>"
                f"<td>{pval_s}</td>"
                f"<td>{'oui' if r.get('significant_5pct') else 'non'}</td></tr>"
            )
        dm_interp = "".join(
            f"<li>{html.escape(str(r['interpretation']))}</li>" for _, r in dm_df.iterrows()
        )
        dm_section = f"""
      <p>Comparaison des pertes au niveau de chaque point prédit (test, MW).
      <code>d = L(a) − L(b)</code> ; p &lt; 0,05 : différence significative à 5 %.</p>
      <table>
        <tr><th>Modèle A</th><th>Modèle B</th><th>Loss</th><th>DM</th><th>p-valeur</th><th>Sig. 5 %</th></tr>
        {dm_rows}
      </table>
      <ul>{dm_interp}</ul>
        """
    else:
        dm_section = "<p><em>Lancer <code>diebold_mariano.py</code> pour remplir <code>results/dm_results.csv</code>.</em></p>"

    attn_section = ""
    for pattern, caption in (
        ("attn_heatmap_window_*.png", "Exemple de poids d'attention sur une fenêtre test"),
        ("attn_heatmap_mean_*windows.png", "Poids d'attention moyens sur plusieurs fenêtres"),
    ):
        matches = sorted(plots_dir.glob(pattern))
        if matches:
            attn_section += img_block(rel_plot(matches[0].name), caption)
    if not attn_section:
        attn_section = "<p><em>Lancer <code>plot_attention_heatmap.py</code> pour le modèle lstm_attention.</em></p>"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    test_range = meta.get("test_datetime_range", ["?", "?"])

    doc = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rapport — Prévision charge PJME (LSTM / Bi-LSTM / Attention)</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #fff;
      --text: #1a1d26;
      --muted: #5c6578;
      --accent: #1e5a8a;
      --border: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      line-height: 1.55;
      color: var(--text);
      background: var(--bg);
      margin: 0;
      padding: 2rem 1rem 4rem;
    }}
    .wrap {{ max-width: 920px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; color: var(--accent); margin-bottom: 0.25rem; }}
    h2 {{
      font-size: 1.2rem;
      margin-top: 2rem;
      padding-bottom: 0.35rem;
      border-bottom: 2px solid var(--accent);
    }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
    section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.25rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.5rem 0.75rem;
      text-align: left;
    }}
    th {{ background: #eef3f9; }}
    tr.best {{ background: #e8f4ea; font-weight: 600; }}
    .fig {{ margin: 1.25rem 0; text-align: center; }}
    .fig img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 4px;
    }}
    .fig figcaption {{
      font-size: 0.85rem;
      color: var(--muted);
      margin-top: 0.5rem;
    }}
    ul {{ padding-left: 1.25rem; }}
    code {{ background: #eef3f9; padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.9em; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem 1.5rem;
    }}
    @media (max-width: 640px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Prévision horaire de la charge PJM East (PJME)</h1>
    <p class="meta">Rapport généré automatiquement — {html.escape(generated)}<br />
    Projet RNA — fenêtre 168 h → horizon 24 h — split chronologique 80 / 10 / 10</p>

    <section>
      <h2>1. Contexte et objectif</h2>
      <p>
        Ce projet prédit la consommation électrique horaire de la zone <strong>PJME</strong>
        (PJM East) à partir de l’historique récent et de variables calendaires.
        L’entrée du modèle est une fenêtre de <strong>{meta.get("input_len", 168)} heures</strong> ;
        la sortie est un vecteur de <strong>{meta.get("horizon", 24)} valeurs</strong> (les 24 prochaines heures de charge).
      </p>
      <p>
        Trois architectures sont comparées : <strong>LSTM</strong>, <strong>Bi-LSTM</strong> et
        <strong>LSTM avec attention additive (Bahdanau)</strong>. Les métriques finales sont calculées
        sur le jeu de <strong>test</strong> uniquement ({html.escape(str(test_range[0]))} → {html.escape(str(test_range[1]))}),
        après dénormalisation en mégawatts (MW).
      </p>
    </section>

    <section>
      <h2>2. Données et prétraitement</h2>
      <ul>
        <li>Source : série horaire <code>PJME_hourly.csv</code> (Kaggle PJM Hourly Energy Consumption).</li>
        <li>Nettoyage DST : agrégation des doublons d’horodatage, grille 1 h complète, interpolation linéaire.</li>
        <li>Features : <code>{", ".join(c for c in meta.get("feature_cols", []) if c != meta.get("target_col"))}</code> + cible <code>{meta.get("target_col", "PJME_MW")}</code>.</li>
        <li>Normalisation Min-Max ajustée sur le <strong>train</strong> seulement, puis appliquée au val et test.</li>
        <li>Lignes totales après features : {meta.get("n_rows_total", "—")} ; train / val / test : {meta.get("n_train_rows")} / {meta.get("n_val_rows")} / {meta.get("n_test_rows")}.</li>
      </ul>
    </section>

    <section>
      <h2>3. Hyperparamètres (communs aux trois modèles)</h2>
      <div class="grid-2">
        <div>
          <table>
            <tr><th>Paramètre</th><th>Valeur</th></tr>
            <tr><td>hidden_size</td><td>{model_cfg.get("hidden_size")}</td></tr>
            <tr><td>num_layers</td><td>{model_cfg.get("num_layers")}</td></tr>
            <tr><td>dropout</td><td>{model_cfg.get("dropout")}</td></tr>
            <tr><td>attn_dim (attention)</td><td>{model_cfg.get("attn_dim")}</td></tr>
            <tr><td>input_dim</td><td>{model_cfg.get("input_dim")}</td></tr>
          </table>
        </div>
        <div>
          <table>
            <tr><th>Entraînement</th><th>Valeur</th></tr>
            <tr><td>batch_size</td><td>{train_cfg.get("batch_size")}</td></tr>
            <tr><td>learning rate (Adam)</td><td>{train_cfg.get("lr")}</td></tr>
            <tr><td>max_epochs</td><td>{train_cfg.get("max_epochs")}</td></tr>
            <tr><td>early stopping (patience)</td><td>{train_cfg.get("patience")}</td></tr>
            <tr><td>device</td><td>{cfg.get("device", "cpu")}</td></tr>
            <tr><td>seed</td><td>{train_cfg.get("seed")}</td></tr>
          </table>
        </div>
      </div>
      <p style="margin-top:1rem;"><strong>Loss :</strong> MSE sur les 24 pas de sortie (charge normalisée pendant l’entraînement).</p>
    </section>

    <section>
      <h2>4. Architectures (résumé)</h2>
      <ul>
        <li><strong>LSTM</strong> : encode la fenêtre de 168 pas ; la dernière sortie temporelle alimente une couche linéaire vers 24 prédictions.</li>
        <li><strong>Bi-LSTM</strong> : deux LSTM (sens direct et inverse) ; concaténation au dernier pas, puis tête linéaire 24 h.</li>
        <li><strong>LSTM + attention</strong> : encodeur LSTM + scores d’attention softmax sur les 168 pas (Bahdanau) ; contexte + état final → 24 h.</li>
      </ul>
    </section>

    <section>
      <h2>5. Résultats sur le jeu de test (MW)</h2>
      <p>Meilleur modèle sur ce run (MAE la plus faible) : <strong>{html.escape(best_name)}</strong>.</p>
      <table>
        <tr>
          <th>Modèle</th>
          <th>MAE (MW)</th>
          <th>RMSE (MW)</th>
          <th>MAPE (%)</th>
        </tr>
        {metrics_rows_html}
      </table>
      <p class="meta" style="margin-top:0.75rem;">
        Source : <code>results/metrics.csv</code> — évaluation complète ({int(metrics_df["n_points"].iloc[0]) if len(metrics_df) else 0} points horaires agrégés).
      </p>
    </section>

    <section>
      <h2>6. Figures d’évaluation (test)</h2>
      {eval_figures if eval_figures else "<p><em>Aucune figure eval_best_* trouvée. Lancer evaluate.py pour chaque modèle.</em></p>"}
    </section>

    <section>
      <h2>7. Courbes d’apprentissage (train / validation)</h2>
      {loss_section if loss_section else "<p><em>Lancer plot_training_history.py pour générer loss_*.png.</em></p>"}
    </section>

    <section>
      <h2>8. Test de Diebold–Mariano</h2>
      {dm_section}
    </section>

    <section>
      <h2>9. Interprétation de l’attention (LSTM + Bahdanau)</h2>
      {attn_section}
    </section>

    <section>
      <h2>10. Discussion</h2>
      <p>
        Les trois modèles obtiennent des erreurs du même ordre de grandeur sur le test.
        Le <strong>{html.escape(best_name)}</strong> affiche légèrement les meilleures métriques dans cette configuration
        (hidden_size={model_cfg.get("hidden_size")}, pas de recherche systématique d’hyperparamètres).
      </p>
      <p>
        Les tests DM indiquent si les écarts de performance entre paires de modèles sont
        statistiquement distinguables au seuil de 5 %, au-delà de la seule comparaison des moyennes (MAE/RMSE).
      </p>
    </section>

    <section>
      <h2>11. Reproductibilité</h2>
      <pre style="background:#eef3f9;padding:1rem;border-radius:6px;overflow:auto;font-size:0.85rem;">python energy_forecast/src/preprocessing.py --all
python energy_forecast/src/train.py --config energy_forecast/config.yaml
python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml
python energy_forecast/src/plot_training_history.py --config energy_forecast/config.yaml
python energy_forecast/src/diebold_mariano.py --config energy_forecast/config.yaml
python energy_forecast/src/plot_attention_heatmap.py --config energy_forecast/config.yaml
python energy_forecast/src/generate_report.py --config energy_forecast/config.yaml</pre>
    </section>
  </div>
</body>
</html>
"""

    out_path.write_text(doc, encoding="utf-8")
    print("Rapport :", out_path.resolve())


if __name__ == "__main__":
    main()
