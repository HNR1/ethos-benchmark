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
        **kwargs,
    ):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.ICU_DISCHARGE] + self.stop_stokens
        
        adm_indices = self._get_indices_of_stokens_with_offset(ST.ICU_ADMISSION)
        # adm_indices = self._get_indices_of_stokens_with_offset(ST.ICU_ADMISSION, timedelta(hours=12))
        # print(len(adm_indices), adm_indices[0].item())
        dc_indices  = self._get_indices_of_stokens(ST.ICU_DISCHARGE)
        dth_indices = self._get_indices_of_stokens(ST.DEATH)
        disc_b4_adm = dc_indices <= adm_indices
        adm_indices = adm_indices[~disc_b4_adm]
        dc_indices  = dc_indices[~disc_b4_adm]
        # TODO: death_b4_dc

        self.start_indices   = adm_indices
        self.outcome_indices = dc_indices
        
        assert len(self.start_indices) == len(self.outcome_indices), f"Found {len(self.start_indices)} admissions and {len(self.outcome_indices)} discharges."
        print(f"Found {len(self.start_indices)} hospital stays with an outcome token.")

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        start_idx = self.start_indices[idx]
        outcome_idx = self.outcome_indices[idx]
        y = {
            # "expected": self.vocab.decode(self.tokens[outcome_idx]),
            "true_token_dist": (outcome_idx - start_idx).item(),
            "true_token_time": (self.times[outcome_idx] - self.times[start_idx]).item(),
            "patient_id": self.patient_id_at_idx[start_idx].item(),
            "prediction_time": self.times[start_idx].item(),
            "data_idx": start_idx.item(),
        }
        if self.is_mimic:
            y["hadm_id"] = self._get_hadm_id(start_idx)
            y["icu_stay_id"] = self._get_icu_stay_id(start_idx)

        return super().__getitem__(start_idx), y

