"""
build_graph.py
---------------
Construiește un graf eterogen (PyTorch Geometric HeteroData) din
patients.csv + episodes.csv.

Schema grafului:
  Noduri:
    - 'patient': un nod per pacient           (features: demografice)
    - 'episode': un nod per episod/măsurătoare (features: parametri fiziologici)

  Muchii:
    - ('patient', 'has', 'episode')      : pacientul -> episoadele lui
    - ('episode', 'rev_has', 'patient')  : inversa, necesară pt. message passing
    - ('episode', 'next', 'episode')     : episodul curent -> episodul următor
                                            al ACELUIAȘI pacient (ordine temporală)
    - ('episode', 'rev_next', 'episode') : inversa relației 'next'

  Task: clasificare de nod pe 'episode' -> predicția 'is_anomaly'.

De ce graf eterogen: pacientul aduce context (comorbidități) care nu se
schimbă între episoade, iar relația 'next' capturează dinamica temporală
fără să avem nevoie de un model secvențial separat -- GraphSAGE va agrega
informație atât de la pacient, cât și de la episoadele vecine temporal.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def build_patient_features(patients: pd.DataFrame) -> torch.Tensor:
    """Normalizează + encodează features de pacient -> tensor [n_patients, F]."""
    age = (patients["age"] - patients["age"].mean()) / patients["age"].std()
    sex_female = (patients["sex"] == "F").astype(float)
    feats = np.stack(
        [
            age.values,
            sex_female.values,
            patients["has_hypertension"].values,
            patients["has_diabetes"].values,
            patients["has_cardiac_history"].values,
        ],
        axis=1,
    )
    return torch.tensor(feats, dtype=torch.float)


def build_episode_features(episodes: pd.DataFrame) -> torch.Tensor:
    """Normalizează parametrii fiziologici -> tensor [n_episodes, F]."""
    cols = ["systolic_bp", "diastolic_bp", "heart_rate", "spo2", "glucose"]
    vals = episodes[cols].values.astype(float)
    mean = vals.mean(axis=0, keepdims=True)
    std = vals.std(axis=0, keepdims=True) + 1e-6
    normed = (vals - mean) / std
    return torch.tensor(normed, dtype=torch.float)


def build_hetero_graph(patients: pd.DataFrame, episodes: pd.DataFrame) -> HeteroData:
    data = HeteroData()

    # ---- noduri ----
    data["patient"].x = build_patient_features(patients)
    data["episode"].x = build_episode_features(episodes)
    data["episode"].y = torch.tensor(episodes["is_anomaly"].values, dtype=torch.long)

    # ---- muchii: patient -> episode ----
    src_patient = episodes["patient_id"].values
    dst_episode = episodes["episode_id"].values
    edge_has = torch.tensor(np.stack([src_patient, dst_episode]), dtype=torch.long)
    data["patient", "has", "episode"].edge_index = edge_has
    data["episode", "rev_has", "patient"].edge_index = edge_has.flip(0)

    # ---- muchii: episode -> next episode (aceeași pacient, ordine temporală) ----
    episodes_sorted = episodes.sort_values(["patient_id", "episode_index"])
    src_next, dst_next = [], []
    for _, group in episodes_sorted.groupby("patient_id"):
        ids = group["episode_id"].values
        if len(ids) > 1:
            src_next.extend(ids[:-1])
            dst_next.extend(ids[1:])
    edge_next = torch.tensor([src_next, dst_next], dtype=torch.long)
    data["episode", "next", "episode"].edge_index = edge_next
    data["episode", "rev_next", "episode"].edge_index = edge_next.flip(0)

    return data


def make_patient_level_split(episodes: pd.DataFrame, patients: pd.DataFrame,
                              train_frac=0.7, val_frac=0.15, seed=42):
    """
    IMPORTANT: split-ul se face la nivel de PACIENT, nu de episod.
    Dacă am amesteca episoade ale aceluiași pacient între train/val/test,
    modelul ar "memora" pacientul în loc să generalizeze -> scurgere de date.
    Returnează masks booleene aliniate cu ordinea nodurilor 'episode'.
    """
    rng = np.random.default_rng(seed)
    patient_ids = patients["patient_id"].values.copy()
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val:])

    # episodes e deja ordonat după episode_id (0..N-1) din generate_data.py,
    # deci indexul din DataFrame corespunde direct indexului nodului în graf
    assert (episodes["episode_id"].values == np.arange(len(episodes))).all(), \
        "episode_id trebuie sa fie 0..N-1 in ordine, altfel indexarea nu se aliniaza"

    train_mask = torch.tensor(episodes["patient_id"].isin(train_patients).values)
    val_mask = torch.tensor(episodes["patient_id"].isin(val_patients).values)
    test_mask = torch.tensor(episodes["patient_id"].isin(test_patients).values)

    return train_mask, val_mask, test_mask


def main():
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    episodes = pd.read_csv(DATA_DIR / "episodes.csv")

    data = build_hetero_graph(patients, episodes)
    train_mask, val_mask, test_mask = make_patient_level_split(episodes, patients)

    data["episode"].train_mask = train_mask
    data["episode"].val_mask = val_mask
    data["episode"].test_mask = test_mask

    print(data)
    print(f"Train episoade: {train_mask.sum().item()}  "
          f"Val: {val_mask.sum().item()}  Test: {test_mask.sum().item()}")
    print(f"Rată anomalii - train: {data['episode'].y[train_mask].float().mean():.3f}  "
          f"val: {data['episode'].y[val_mask].float().mean():.3f}  "
          f"test: {data['episode'].y[test_mask].float().mean():.3f}")

    torch.save(data, DATA_DIR / "hetero_graph.pt")
    print(f"Graf salvat în: {(DATA_DIR / 'hetero_graph.pt').resolve()}")


if __name__ == "__main__":
    main()
