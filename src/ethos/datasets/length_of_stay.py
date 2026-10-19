import abc
from pathlib import Path

import torch as th
from datetime import timedelta

from ..constants import SpecialToken as ST
from .base import InferenceDataset


class LosPredictionDataset(InferenceDataset, abc.ABC):
    def __init__(
        self,
        input_dir: str | Path,
        n_positions: int = 2048,
        offset: timedelta = timedelta(),
        **kwargs,
    ):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ICU_DISCHARGE] + self.stop_stokens
        self.offset = offset
        
        adm_indices = self._get_indices_of_stokens(ST.ICU_ADMISSION)
        start_indices = self._get_indices_of_stokens_with_offset(ST.ICU_ADMISSION, self.offset)
        assert (start_indices >= 0).all(), f"Found no start indices for offset {self.offset}."
        dc_indices  = self._get_indices_of_stokens(ST.ICU_DISCHARGE)
        dth_indices = self._get_indices_of_stokens(ST.DEATH)
        
        dc_b4_start = dc_indices <= start_indices
        dth_b4_start = death_b4_disc = (
            (dth_indices[:, None] > adm_indices[None, :]) &
            (dth_indices[:, None] <= start_indices[None, :])
        ).any(dim=0)
        valid_indices = ~(dc_b4_start | dth_b4_start)
        
        adm_indices = adm_indices[valid_indices]
        start_indices = start_indices[valid_indices]
        dc_indices  = dc_indices[valid_indices]
        dc_or_dth_indices = th.tensor(list(sorted(set(dc_indices).union(set(dth_indices)))))

        self.adm_indices = adm_indices
        self.start_indices = start_indices 
        self.outcome_indices = self._match(dc_or_dth_indices, adm_indices)
        
        assert len(self.start_indices) == len(self.outcome_indices) == len(self.adm_indices), f"Found {len(self.start_indices)} admissions and {len(self.outcome_indices)} discharges."
        print(f"Found {len(self.start_indices)} hospital stays with an outcome token.")

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        adm_idx = self.adm_indices[idx]
        start_idx = self.start_indices[idx]
        outcome_idx = self.outcome_indices[idx]
        y = {
            "expected": self.vocab.decode(self.tokens[outcome_idx]),
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "patient_id": self.patient_id_at_idx[adm_idx].item(),
            "prediction_time": self.times[adm_idx].item() + self.offset.total_seconds() * 1e6,
            "data_idx": start_idx.item(),
        }
        if self.is_mimic:
            y["hadm_id"] = self._get_hadm_id(adm_idx)
            y["icu_stay_id"] = self._get_icu_stay_id(adm_idx)

        return super().__getitem__(start_idx), y

