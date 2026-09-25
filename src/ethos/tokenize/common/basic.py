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
            | (pl.col("time") <= pl.col("death_time"))
        )
        .drop("death_time")
        .sort(["subject_id", "time"], nulls_last=False)
    )


def remove_admissions_after_death(df: pl.DataFrame) -> pl.DataFrame:
    # Get the earliest death time for each patient
    death_times = (
        df.filter(pl.col("code") == "MEDS_DEATH")
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("death_time"))
    )

    # Find admissions occurring at or after death
    invalid_hadm_ids = (
        df.filter(pl.col("code").str.contains("HOSPITAL_ADMISSION"))
        .join(death_times, on="subject_id", how="inner")
        .filter(pl.col("time") >= pl.col("death_time"))
        .select(["subject_id", "hadm_id"])
        .unique()
    )

    # Remove all rows belonging to those admissions
    return (
        df.join(
            invalid_hadm_ids,
            on=["subject_id", "hadm_id"],
            how="anti",
        )
        .sort(["subject_id", "time"], nulls_last=False)
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
