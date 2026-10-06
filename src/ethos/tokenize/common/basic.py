import pickle
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ...constants import STATIC_DATA_FN
from ...constants import SpecialToken as ST
from ...vocabulary import Vocabulary
from ..patterns import MatchAndRevise, ScanAndAggregate
from ..utils import create_prefix_or_chain, static_class

# -------------------- NEW METHODS --------------------
def remove_rows_after_death(df: pl.DataFrame) -> pl.DataFrame:
    death_times = (
        df.filter(pl.col("code") == "MEDS_DEATH")
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("death_time"))
    )

    return (
        df.join(death_times, on="subject_id", how="left")
        .filter(
            pl.col("death_time").is_null()
            | pl.col("time").is_null()
            | (pl.col("time") <= pl.col("death_time"))
            | pl.col("code").str.contains(r"^(HOSPITAL_|ICU_|DRG)")
        )
        .drop("death_time")
        .sort(["subject_id", "time"], nulls_last=False)
    )


def remove_post_death_stays(df: pl.DataFrame) -> pl.DataFrame:
    df = _remove_stays_after_death(df, type='ICU')
    df = _remove_stays_after_death(df, type='HOSP')
    return df

    
def _remove_stays_after_death(df: pl.DataFrame, type: str = 'HOSP') -> pl.DataFrame:
    if type == 'HOSP':
        admission_code = "HOSPITAL_ADMISSION"
        id_col = "hadm_id"
    elif type == 'ICU':
        admission_code = "ICU_ADMISSION"
        id_col = "icustay_id"
    else:
        raise ValueError(f"Invalid type '{type}'. Must be 'HOSP' or 'ICU'.")
    
    # Get the earliest death time for each patient
    death_times = (
        df.filter(pl.col("code") == "MEDS_DEATH")
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("death_time"))
    )

    # Find admissions occurring at or after death
    invalid_hadm_ids = (
        df.filter(pl.col("code").str.contains(admission_code))
        .join(death_times, on="subject_id", how="inner")
        .filter(pl.col("time") >= pl.col("death_time"))
        .select(["subject_id", id_col])
        .unique()
    )

    # Remove all rows belonging to those admissions
    return (
        df.join(
            invalid_hadm_ids,
            on=["subject_id", id_col],
            how="anti",
        )
        .sort(
            ["subject_id", "time"], 
            nulls_last=False,
        )
    )


def align_admissions_and_discharges(df: pl.DataFrame) -> pl.DataFrame:
    df = _align_admission_time_to_first_event(df, type='ICU')
    df = _align_discharge_time_to_last_event(df, type='ICU')
    df = _align_admission_time_to_first_event(df, type='HOSP')
    df = _align_discharge_time_to_last_event(df, type='HOSP')
    return df


def _align_admission_time_to_first_event(df: pl.DataFrame, type: str = 'HOSP') -> pl.DataFrame:
    if type == 'HOSP':
        admission_code = "HOSPITAL_ADMISSION"
        id_col = "hadm_id"
    elif type == 'ICU':
        admission_code = "ICU_ADMISSION"
        id_col = "icustay_id"
    else:
        raise ValueError(f"Invalid type '{type}'. Must be 'HOSP' or 'ICU'.")
    
    df = df.with_row_index("_row_id")

    # Find the earliest timed event for each hospital stay
    first_event_times = (
        df
        .filter(
            pl.col(id_col).is_not_null()
            & pl.col("time").is_not_null()
            & ~pl.col("code").str.contains('ICD//')
            & ~pl.col("code").str.contains('MEDICATION//')
            & ~pl.col("code").str.contains('HCPCS//')
            & ~pl.col("code").str.contains('TRANSFER_TO//ED//')
            & ~pl.col("code").str.contains('ED_')
        )
        .group_by(id_col)
        .agg(
            pl.col("time").min().alias("first_event_time")
        )
    )

    # Find admission events and their current times
    admission_updates = (
        df
        .filter(
            pl.col("code").str.contains(admission_code)
            & pl.col(id_col).is_not_null()
            & pl.col("time").is_not_null()
        )
        .select(
            "_row_id",
            id_col,
            pl.col("time").alias("admission_time"),
        )
        .join(
            first_event_times,
            on=id_col,
            how="left",
        )
        # Only move the admission if it is later than the first event of the hospital stay
        .filter(
            pl.col("admission_time") > pl.col("first_event_time")
        )
        .select(
            "_row_id",
            pl.col("first_event_time").alias("new_time"),
        )
    )

    return (
        df
        .join(
            admission_updates,
            on="_row_id",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("new_time").is_not_null())
            .then(pl.col("new_time"))
            .otherwise(pl.col("time"))
            .alias("time")
        )
        .drop(["_row_id", "new_time"])
        .sort(
            ["subject_id", "time", "hadm_id", "icustay_id"],
            nulls_last=False,
        )
        .with_row_index()
        .drop("index")
    )


def _align_discharge_time_to_last_event(df: pl.DataFrame, type: str = 'HOSP') -> pl.DataFrame:
    if type == 'HOSP':
        discharge_code = "HOSPITAL_DISCHARGE"
        id_col = "hadm_id"
    elif type == 'ICU':
        discharge_code = "ICU_DISCHARGE"
        id_col = "icustay_id"
    else:
        raise ValueError(f"Invalid type '{type}'. Must be 'HOSP' or 'ICU'.")

    df = df.with_row_index("_row_id")

    # Find the latest timed event for each admission
    last_event_times = (
        df
        .filter(
            pl.col(id_col).is_not_null()
            & pl.col("time").is_not_null()
            & ~pl.col("code").str.contains('ICD//')
            & ~pl.col("code").str.contains('MEDICATION//')
            & ~pl.col("code").str.contains('TRANSFER_TO//')
        )
        .group_by(id_col)
        .agg(
            pl.col("time").max().alias("last_event_time")
        )
    )

    # Get discharge rows and their current times
    discharge_updates = (
        df
        .filter(
            pl.col("code").str.contains(discharge_code)
            & pl.col(id_col).is_not_null()
            & pl.col("time").is_not_null()
        )
        .select(
            "_row_id",
            id_col,
            pl.col("time").alias("discharge_time"),
        )
        .join(
            last_event_times,
            on=id_col,
            how="left",
        )
        # Only update if the last event occurred after the discharge
        .filter(
            pl.col("last_event_time") > pl.col("discharge_time")
        )
        # .select(
        #     "_row_id",
        #     pl.col("last_event_time").alias("new_time"),
        # )
    )
    
    rows_to_update = (
        df
        .join(
            discharge_updates,
            on=id_col,
            how="inner",
        )
        .filter(
            (
                pl.col("code").str.contains(discharge_code)
                | pl.col("code").str.contains("DRG//")
            )
            & (pl.col("time") == pl.col("discharge_time"))
        )
        .select(
            "_row_id",
            pl.col("last_event_time").alias("new_time"),
        )
    )

    return (
        df
        .join(
            rows_to_update,
            on="_row_id",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("new_time").is_not_null())
            .then(pl.col("new_time"))
            .otherwise(pl.col("time"))
            .alias("time")
        )
        .with_columns(
            pl.when(pl.col("code").str.contains(discharge_code))
            .then(0)
            .when(pl.col("code").str.contains("DRG//"))
            .then(1)
            .otherwise(2)
            .alias("_sort_order")
        )
        .drop(["_row_id", "new_time"])
        .sort(
            ["subject_id", "time", "_sort_order", "hadm_id", "icustay_id"],
            nulls_last=False,
        )
        .drop("_sort_order")
        .with_row_index()
        .drop("index")
    )


def align_death_time_to_next_discharge(df: pl.DataFrame) -> pl.DataFrame:
    # Give every original row a unique ID
    df = df.with_row_index("_row_id")

    # Death events
    deaths = (
        df
        .filter(pl.col("code") == "MEDS_DEATH")
        .select(
            "_row_id",
            "subject_id",
            pl.col("time").alias("death_time"),
        )
    )

    # All discharge events
    discharges = (
        df
        .filter(pl.col("code").str.contains("HOSPITAL_DISCHARGE"))
        .select(
            "subject_id",
            pl.col("time").alias("discharge_time"),
        )
    )

    # For every death, find the earliest discharge strictly after it
    death_updates = (
        deaths
        .join(discharges, on="subject_id", how="left")
        .filter(pl.col("discharge_time") > pl.col("death_time"))
        .group_by("_row_id")
        .agg(
            pl.col("discharge_time").min().alias("new_time")
        )
    )

    # Update only death rows that have a subsequent discharge
    return (
        df
        .join(death_updates, on="_row_id", how="left")
        .with_columns(
            pl.when(pl.col("new_time").is_not_null())
            .then(pl.col("new_time"))
            .otherwise(pl.col("time"))
            .alias("time")
        )
        .drop(["_row_id", "new_time"])
        .sort(["subject_id", "time"], nulls_last=False)
        .with_row_index()
        .drop("index")
    )


def manual_fixes(df: pl.DataFrame) -> pl.DataFrame:
    target_time = pl.datetime(2173, 9, 23, 0, 0, 0)

    return (
        df
        # Set the time for the specific target event
        .with_columns(
                pl.when(
                    (pl.col("hadm_id") == 28241666.0)
                    & (pl.col("code").str.contains("HOSPITAL_ADMISSION"))
                )
                .then(target_time)
                .otherwise(pl.col("time"))
                .alias("time")
            )
            # Temporary key to control ordering of events at the same timestamp
            .with_columns(
                pl.when(pl.col("code").str.contains("HOSPITAL_DISCHARGE"))
                .then(0)
                .when(pl.col("code").str.contains("HOSPITAL_ADMISSION"))
                .then(1)
                .otherwise(0)
                .alias("_event_order")
            )
        # Sort discharge before admission when subject_id + time are equal
        .sort(
            [
                "subject_id",
                "time",
                "_event_order",
                "hadm_id",
                "icustay_id",
            ],
            nulls_last=False,
        )
        .drop("_event_order")
        .with_row_index()
        .drop("index")
    )

# -------------------- OG METHODS --------------------
def filter_codes(
    df: pl.DataFrame, *, codes_to_remove: Sequence[str], is_prefix: bool = False
) -> pl.DataFrame:
    expr = pl.col("code").cast(str).is_in(codes_to_remove)
    if is_prefix:
        expr = create_prefix_or_chain(codes_to_remove)
    return df.filter(~expr)


def apply_vocab(df: pl.DataFrame, *, vocab: str | list[str] | None = None) -> pl.DataFrame:
    if vocab is None:
        return df
    elif isinstance(vocab, str):
        vocab = list(Vocabulary.from_path(vocab))
    return df.filter(pl.col("code").is_in(vocab))


@static_class
class CodeCounter(ScanAndAggregate):
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.col("code").value_counts()).unnest("code")

    def agg(self, in_fps: list, out_fp: str | Path) -> None:
        dfs = [pl.scan_parquet(fp) for fp in in_fps]
        df = dfs[0]
        for rdf in dfs[1:]:
            df = df.join(rdf, on="code", how="full", coalesce=True, join_nulls=True).select(
                "code", pl.sum_horizontal(pl.exclude("code"))
            )
        df.sort("count", descending=True).collect().write_csv(out_fp)


@static_class
class StaticDataCollector(ScanAndAggregate):
    patient_id_col = MatchAndRevise.sort_cols[0]

    def __call__(self, df: pl.DataFrame, *, static_code_prefixes: list[str]) -> pl.DataFrame:
        df = (
            df.select(self.patient_id_col, "code", pl.col("time").cast(pl.Int64))
            .filter(create_prefix_or_chain(static_code_prefixes))
            .group_by(
                self.patient_id_col,
                prefix=pl.col("code").str.split("//").list.get(0),
            )
            .agg("code", "time")
            .with_columns(pl.struct(code="code", time="time"))
            .pivot(index=self.patient_id_col, on="prefix", values="code")
            .with_columns(
                pl.when(pl.col(col_name).struct[0].is_null())
                .then(pl.struct(code=pl.lit([f"{col_name}//UNKNOWN"])))
                .otherwise(col_name)
                .alias(col_name)
                for col_name in static_code_prefixes
                if col_name != ST.DOB
            )
        )
        # maintain the order of columns, so that the output is deterministic
        return df.select(sorted(df.columns))

    def agg(self, in_fps: list, out_fp: str | Path) -> None:
        # TODO: Let's store it in parquet instead of pickle
        df = pl.read_parquet(in_fps)
        out_dict = df.rows_by_key(self.patient_id_col, named=True, unique=True)
        with Path(out_fp).with_name(STATIC_DATA_FN).open("wb") as f:
            pickle.dump(out_dict, f)
