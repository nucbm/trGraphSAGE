# GraphSAGE pentru detecția anomaliilor fiziologice — proiect demonstrativ

Pipeline complet, pe date **sintetice**, pentru: generare date → construcție
graf eterogen → antrenare GraphSAGE → evaluare. Scopul e să aveți un schelet
funcțional pe care să îl adaptați apoi la date reale.

## Instalare

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 1. PyTorch (CPU sau CUDA, în funcție de mașina voastră)
#    Pentru varianta CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
#    Pentru GPU (CUDA 12.1, exemplu):
#    pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. PyTorch Geometric
pip install torch_geometric

# 3. Restul dependențelor
pip install pandas numpy scikit-learn
```

> Recomandare: dacă nu aveți GPU local, e mai simplu să rulați totul într-un
> notebook Google Colab (are torch + CUDA preinstalate) — doar copiați cele
> 4 fișiere `.py` acolo.

## Rulare

```bash
python generate_data.py   # -> data/patients.csv, data/episodes.csv
python build_graph.py     # -> data/hetero_graph.pt
python train.py           # antrenează modelul, afișează metrici, salvează model_best.pt
```

## Ce conține fiecare fișier

| Fișier | Rol |
|---|---|
| `generate_data.py` | Generează pacienți + episoade de măsurători sintetice, cu anomalii injectate după reguli clinice simplificate |
| `build_graph.py` | Transformă CSV-urile într-un `HeteroData` (PyG): noduri `patient`/`episode`, muchii `has`/`next` + inversele lor; face split-ul train/val/test **la nivel de pacient** |
| `model.py` | `HeteroGraphSAGE` — GraphSAGE heterogen, cu `SAGEConv` per tip de muchie, agregate prin `HeteroConv` |
| `train.py` | Buclă de antrenare full-batch, cu ponderare de clasă (anomaliile sunt rare), early stopping pe AUPRC, evaluare AUROC/AUPRC/F1 |

## Decizii de proiectare importante (și de ce)

1. **Split pe pacient, nu pe episod** — altfel episoade ale aceluiași
   pacient ajung și în train și în test, iar modelul "memorează" pacientul
   în loc să generalizeze.

2. **AUPRC/AUROC/F1, nu accuracy** — anomaliile sunt ~8-9% din date;
   accuracy ar fi înșelător (un model care prezice mereu "normal" ar avea
   accuracy >90%).

3. **Ponderare de clasă în loss** — calculată *doar* pe train, ca să nu
   introducem scurgere de informație din val/test.

4. **Muchii `next`/`rev_next`** — capturează ordinea temporală a
   episoadelor aceluiași pacient, fără a avea nevoie de un model
   secvențial separat (LSTM etc.) — GraphSAGE agregă acest context direct.

5. **Full-batch aici, dar cu exemplu de `NeighborLoader` în `train.py`** —
   pe un graf real, mult mai mare, treceți la mini-batching cu sampling
   de vecini (asta e ideea centrală a GraphSAGE, spre deosebire de GCN).

## Trecerea la date reale

Când înlocuiți datele sintetice cu date reale:
- Păstrați aceeași structură de fișiere (`patients.csv`, `episodes.csv`)
  cu coloanele echivalente, și restul pipeline-ului rămâne valabil.

- Recalculați `compute_class_weights` — rata reală de anomalii poate
  diferi mult de cea sintetică.

- Verificați valorile lipsă (foarte frecvente în date clinice reale) —
  `build_episode_features` presupune momentan date complete.

- Revizuiți schema grafului: dacă aveți mai multe tipuri de evenimente
  (nu doar unul), sau relații suplimentare (secție, medic, diagnostic),
  adăugați-le ca noduri/muchii noi în `build_graph.py`.
