"""
generate_data.py
-----------------
Generează un set de date SINTETIC (nu real) care simulează:
  - pacienți cu demografice + comorbidități
  - episoade (evenimente) de măsurare a parametrilor fiziologici
    (tensiune arterială sistolică/diastolică, puls, SpO2, glicemie)
  - un flag de "anomalie" per episod, generat pe baza unor reguli
    clinice simplificate + zgomot aleator

Scopul: să avem date tabulare plauzibile din punct de vedere STRUCTURAL,
pe care să construim graful și pipeline-ul GraphSAGE. Pentru validare
clinică reală, aceste date trebuie înlocuite cu date reale (ex. MIMIC-III/IV).

Output:
  data/patients.csv   -> un rând per pacient
  data/episodes.csv   -> un rând per episod (măsurătoare), ordonat temporal per pacient
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_PATIENTS = 600
MIN_EPISODES_PER_PATIENT = 3
MAX_EPISODES_PER_PATIENT = 20
OUTPUT_DIR = Path(__file__).parent / "data"


def generate_patients(rng: np.random.Generator, n_patients: int) -> pd.DataFrame:
    """Generează demograficele și comorbiditățile de bază per pacient."""
    age = rng.integers(18, 90, size=n_patients)
    sex = rng.choice(["M", "F"], size=n_patients)
    has_hypertension = rng.random(n_patients) < (0.15 + (age > 55) * 0.35)
    has_diabetes = rng.random(n_patients) < (0.10 + (age > 60) * 0.25)
    has_cardiac_history = rng.random(n_patients) < (0.05 + (age > 65) * 0.30)

    patients = pd.DataFrame(
        {
            "patient_id": np.arange(n_patients),
            "age": age,
            "sex": sex,
            "has_hypertension": has_hypertension.astype(int),
            "has_diabetes": has_diabetes.astype(int),
            "has_cardiac_history": has_cardiac_history.astype(int),
        }
    )
    return patients


def generate_episodes(rng: np.random.Generator, patients: pd.DataFrame) -> pd.DataFrame:
    """
    Pentru fiecare pacient, generează o secvență temporală de episoade
    (măsurători). Anomaliile sunt injectate pe baza unor reguli clinice
    simplificate, în funcție de comorbiditățile pacientului, plus zgomot.
    """
    rows = []
    episode_global_id = 0

    for _, patient in patients.iterrows():
        n_episodes = rng.integers(MIN_EPISODES_PER_PATIENT, MAX_EPISODES_PER_PATIENT + 1)

        # bază individuală (fiecare pacient are un "nivel de bază" ușor diferit)
        base_systolic = rng.normal(120, 8) + patient["has_hypertension"] * rng.normal(15, 5)
        base_diastolic = rng.normal(78, 6) + patient["has_hypertension"] * rng.normal(8, 3)
        base_hr = rng.normal(75, 8) + patient["has_cardiac_history"] * rng.normal(10, 4)
        base_spo2 = rng.normal(97, 1.2)
        base_glucose = rng.normal(95, 10) + patient["has_diabetes"] * rng.normal(45, 15)

        t = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 30)))

        for episode_idx in range(n_episodes):
            # timp între episoade: ore -> zile, aleator
            t = t + pd.Timedelta(hours=float(rng.exponential(18)))

            # zgomot de măsurare + posibil "eveniment" acut random
            acute_event = rng.random() < 0.12  # 12% șansă de eveniment acut la acest episod
            shock = rng.normal(0, 1) * (25 if acute_event else 4)

            systolic = base_systolic + rng.normal(0, 6) + shock
            diastolic = base_diastolic + rng.normal(0, 5) + shock * 0.5
            heart_rate = base_hr + rng.normal(0, 6) + abs(shock) * 0.6
            spo2 = np.clip(base_spo2 + rng.normal(0, 0.8) - abs(shock) * 0.05, 70, 100)
            glucose = base_glucose + rng.normal(0, 8) + (shock if patient["has_diabetes"] else 0)

            # regulă simplificată de anomalie clinică (nu e sfat medical, doar
            # o regulă sintetică pentru a avea o etichetă plauzibilă)
            is_anomaly = int(
                systolic > 160
                or systolic < 90
                or diastolic > 100
                or heart_rate > 130
                or heart_rate < 45
                or spo2 < 92
                or glucose > 200
                or glucose < 60
            )
            # puțin zgomot pe etichetă (adnotare clinică imperfectă în realitate)
            if rng.random() < 0.03:
                is_anomaly = 1 - is_anomaly

            rows.append(
                {
                    "episode_id": episode_global_id,
                    "patient_id": patient["patient_id"],
                    "episode_index": episode_idx,  # ordinea în secvența pacientului
                    "timestamp": t,
                    "systolic_bp": round(float(systolic), 1),
                    "diastolic_bp": round(float(diastolic), 1),
                    "heart_rate": round(float(heart_rate), 1),
                    "spo2": round(float(spo2), 1),
                    "glucose": round(float(glucose), 1),
                    "is_anomaly": is_anomaly,
                }
            )
            episode_global_id += 1

    episodes = pd.DataFrame(rows)
    return episodes


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patients = generate_patients(rng, N_PATIENTS)
    episodes = generate_episodes(rng, patients)

    patients.to_csv(OUTPUT_DIR / "patients.csv", index=False)
    episodes.to_csv(OUTPUT_DIR / "episodes.csv", index=False)

    print(f"Pacienți generați: {len(patients)}")
    print(f"Episoade generate: {len(episodes)}")
    print(f"Procent anomalii: {episodes['is_anomaly'].mean():.2%}")
    print(f"Fișiere salvate în: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
