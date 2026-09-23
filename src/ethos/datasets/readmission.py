from datetime import timedelta
from pathlib import Path

import torch as th
import numpy as np

from ..constants import SpecialToken as ST
from .base import InferenceDataset


class ReadmissionDataset(InferenceDataset):
    """Generates timelines that terminate at the DRG (Diagnosis-Related Group) token associated with
    hospital stays.

    The target variable is the patient's next admission.
    """

    time_limit = timedelta(days=30)

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ADMISSION] + self.stop_stokens

        dc_indices = self._get_indices_of_stokens(ST.DISCHARGE)
        # Remove cases when the patient died in the hospital, so dicharge is not preceded by death
        death_indices = self._get_indices_of_stokens(ST.DEATH)
        dc_indices = dc_indices[~th.isin(dc_indices - 1, death_indices)]
        print(f"Found {len(dc_indices)} discharge tokens after removing those preceded by death.")

        adm_or_death_indices = self._get_indices_of_stokens(
            [ST.ADMISSION, ST.DEATH, ST.TIMELINE_END]
        )
        self.outcome_indices = self._match(adm_or_death_indices, dc_indices)

        drg_indices = self._get_indices_of_stokens(
            [stoken for stoken in self.vocab if stoken.startswith("DRG//")]
        )
        self.start_indices = self._match(drg_indices, dc_indices)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        outcome_idx = self.outcome_indices[idx]

        y = {
            "expected": self.vocab.decode(self.tokens[outcome_idx]),
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "patient_id": self.patient_id_at_idx[start_idx].item(),
            "prediction_time": self.times[start_idx].item(),
            "data_idx": start_idx.item(),
        }

        if self.is_mimic:
            y["hadm_id"] = self._get_hadm_id(start_idx)

        return super().__getitem__(start_idx), y
    

class ReadmissionDatasetNew(InferenceDataset):
    """Generates timelines that terminate at the DRG (Diagnosis-Related Group) token associated with
    hospital stays.

    The target variable is the patient's next admission.
    """

    time_limit = timedelta(days=30)

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ADMISSION] + self.stop_stokens
        
        adm_indices = self._get_indices_of_stokens(ST.ADMISSION)
        dc_indices = self._get_indices_of_stokens(ST.DISCHARGE)
        death_indices = self._get_indices_of_stokens(ST.DEATH)
        
        print(f"Found {len(dc_indices)} discharge tokens before removing those preceded by death.")
        # First condition: discharge must occur after admission
        valid_interval = dc_indices > adm_indices

        # Find which admission interval each death belongs to
        interval_idx = th.searchsorted(adm_indices, death_indices) - 1

        # Only consider deaths that could belong to an interval
        valid_death = interval_idx >= 0

        # Make sure the death occurs before the corresponding discharge
        safe_idx = interval_idx.clamp(min=0)
        valid_death &= death_indices < dc_indices[safe_idx]

        # Mark intervals that contain a death
        has_death = th.zeros(
            len(adm_indices),
            dtype=th.bool,
            device=adm_indices.device
        )
        has_death[interval_idx[valid_death]] = True

        # Both conditions must hold
        keep = valid_interval & ~has_death

        dc_indices = dc_indices[keep]
        print(f"Found {len(dc_indices)} discharge tokens after removing those preceded by death.")
        self.dc_indices = dc_indices

        adm_or_death_indices = self._get_indices_of_stokens(
            [ST.ADMISSION, ST.DEATH, ST.TIMELINE_END]
        )
        self.outcome_indices = self._match(adm_or_death_indices, dc_indices)

        drg_indices = self._get_indices_of_stokens(
            [stoken for stoken in self.vocab if stoken.startswith("DRG//")]
        )
        self.start_indices = self._match(drg_indices, dc_indices)


    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        outcome_idx = self.outcome_indices[idx]

        y = {
            "expected": self.vocab.decode(self.tokens[outcome_idx]),
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "patient_id": self.patient_id_at_idx[start_idx].item(),
            "prediction_time": self.times[start_idx].item(),
            "data_idx": start_idx.item(),
        }

        if self.is_mimic:
            y["hadm_id"] = self._get_hadm_id(start_idx)

        return super().__getitem__(start_idx), y

