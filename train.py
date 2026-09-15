"""
train.py
--------
Antrenează HeteroGraphSAGE pentru clasificarea nodurilor 'episode'
(anomalie fiziologică: da/nu).

Notă despre scalabilitate: graful nostru sintetic e mic (~600 pacienți,
~7000 episoade), deci antrenăm full-batch (tot graful într-un singur
forward pass). Pentru un graf real, mult mai mare, ar trebui trecut la
mini-batching cu torch_geometric.loader.NeighborLoader / HGTLoader,
care fac exact sampling-ul de vecini care dă numele lui GraphSAGE.
Las un exemplu comentat la final pentru acest caz.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report

from model import HeteroGraphSAGE

DATA_DIR = Path(__file__).parent / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Anomaliile sunt rare (~8-9%) -> fără ponderare, modelul poate învăța
    trivial să prezică mereu 'normal'. Ponderăm invers proporțional cu
    frecvența claselor, calculat DOAR pe train, ca să nu scurgem info.
    """
    y_train = y[mask]
    counts = torch.bincount(y_train, minlength=2).float()
    weights = counts.sum() / (2.0 * counts)
    return weights


@torch.no_grad()
def evaluate(model, data, mask, threshold=0.5):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    probs = F.softmax(out, dim=1)[:, 1]

    y_true = data["episode"].y[mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics, y_true, y_pred


def train(num_epochs=100, hidden_channels=64, num_layers=2, lr=0.005,
          weight_decay=5e-4, dropout=0.3, patience=15):
    data = torch.load(DATA_DIR / "hetero_graph.pt", weights_only=False).to(DEVICE)

    model = HeteroGraphSAGE(
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        out_channels=2,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    # trecere "dummy" ca sa initializam layerele Lazy (-1 din Linear/SAGEConv)
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = data["episode"].train_mask
    val_mask = data["episode"].val_mask
    test_mask = data["episode"].test_mask
    y = data["episode"].y

    class_weights = compute_class_weights(y, train_mask).to(DEVICE)
    print(f"Ponderi de clasă (0=normal, 1=anomalie): {class_weights.tolist()}")

    best_val_auprc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[train_mask], y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            val_metrics, _, _ = evaluate(model, data, val_mask)
            print(
                f"Epoca {epoch:03d} | loss={loss.item():.4f} | "
                f"val_auroc={val_metrics['auroc']:.4f} | "
                f"val_auprc={val_metrics['auprc']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            if val_metrics["auprc"] > best_val_auprc:
                best_val_auprc = val_metrics["auprc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve * 5 >= patience:
                print(f"Early stopping la epoca {epoch} (fără îmbunătățire pe val_auprc).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_true, y_pred = evaluate(model, data, test_mask)
    print("\n=== Rezultate finale pe test ===")
    print(f"AUROC: {test_metrics['auroc']:.4f}")
    print(f"AUPRC: {test_metrics['auprc']:.4f}")
    print(f"F1:    {test_metrics['f1']:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=["normal", "anomalie"], zero_division=0))

    torch.save(model.state_dict(), DATA_DIR / "model_best.pt")
    print(f"Model salvat în: {(DATA_DIR / 'model_best.pt').resolve()}")

    return model, test_metrics


if __name__ == "__main__":
    train()


# ---------------------------------------------------------------------------
# Exemplu (comentat) de mini-batching pentru grafuri MARI, folosind
# NeighborLoader -- de folosit când graful nu mai încape confortabil
# într-un singur forward pass full-batch:
#
# from torch_geometric.loader import NeighborLoader
#
# train_loader = NeighborLoader(
#     data,
#     num_neighbors={key: [15, 10] for key in data.edge_types},  # per layer
#     batch_size=512,
#     input_nodes=("episode", data["episode"].train_mask),
#     shuffle=True,
# )
#
# for batch in train_loader:
#     batch = batch.to(DEVICE)
#     out = model(batch.x_dict, batch.edge_index_dict)
#     loss = F.cross_entropy(out[:batch["episode"].batch_size],
#                             batch["episode"].y[:batch["episode"].batch_size])
#     ...
# ---------------------------------------------------------------------------
