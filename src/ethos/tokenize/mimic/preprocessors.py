import numpy as np
import polars as pl
from datetime import datetime

from ...constants import SpecialToken as ST
from ..patterns import MatchAndRevise
from ..utils import apply_vocab_to_multitoken_codes, unify_code_names


class TableData:
    
    @staticmethod
    def remove_post_death_stays(df: pl.DataFrame) -> pl.DataFrame:
        df = TableData._remove_stays_after_death(df, type='ED')
        df = TableData._remove_stays_after_death(df, type='ICU')
        df = TableData._remove_stays_after_death(df, type='HOSP')
        return df

    @staticmethod
    def _remove_stays_after_death(df: pl.DataFrame, type: str = 'HOSP') -> pl.DataFrame:
        if type == 'HOSP':
            admission_code = "HOSPITAL_ADMISSION"
            id_col = "hadm_id"
        elif type == 'ICU':
            admission_code = "ICU_ADMISSION"
            id_col = "icustay_id"
        elif type == 'ED':
            admission_code = "ED_REGISTRATION"
            id_col = "edstay_id"
        else:
            raise ValueError(f"Invalid type '{type}'. Must be 'HOSP', 'ICU' or 'ED'.")
        
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

    @staticmethod
    def align_admissions_and_discharges(df: pl.DataFrame) -> pl.DataFrame:
        df = TableData._align_discharge_time_to_last_event(df, type='ED')
        df = TableData._align_admission_time_to_first_event(df, type='ICU')
        df = TableData._align_discharge_time_to_last_event(df, type='ICU')
        df = TableData._align_admission_time_to_first_event(df, type='HOSP')
        df = TableData._align_discharge_time_to_last_event(df, type='HOSP')
        return df

    @staticmethod
    def _align_admission_time_to_first_event(df: pl.DataFrame, type: str = 'HOSP') -> pl.DataFrame:
        if type == 'HOSP':
            admission_code = "HOSPITAL_ADMISSION"
            id_col = "hadm_id"
        elif type == 'ICU':
            admission_code = "ICU_ADMISSION"
            id_col = "icustay_id"
        elif type == 'ED':
            admission_code = "ED_REGISTRATION"
            id_col = "edstay_id"
        else:
            raise ValueError(f"Invalid type '{type}'. Must be 'HOSP', 'ICU' or 'ED'.")
        
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
                & ~pl.col("code").str.contains('LAB//')
                & ~pl.col("code").str.contains('_DISCHARGE')
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

    @staticmethod
    def _align_discharge_time_to_last_event(df: pl.DataFrame, type: str = 'HOSP') -> pl.DataFrame:
        if type == 'HOSP':
            discharge_code = "HOSPITAL_DISCHARGE"
            id_col = "hadm_id"
        elif type == 'ICU':
            discharge_code = "ICU_DISCHARGE"
            id_col = "icustay_id"
        elif type == 'ED':
            discharge_code = "ED_OUT"
            id_col = "edstay_id"
        else:
            raise ValueError(f"Invalid type '{type}'. Must be 'HOSP', 'ICU' or 'ED'.")

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
                & ~pl.col("code").str.contains('ED_')
                & ~pl.col("code").str.contains('ICU_')
                & ~pl.col("code").str.contains('HOSPITAL_DISCHARGE')
                & ~pl.col("code").str.contains('DRG//')
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
                (pl.col("code").str.contains(discharge_code)
                | pl.col("code").str.contains("DRG//")
                | pl.col("code").str.contains("DIAGNOSIS//"))
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
                pl.col("last_event_time") >= pl.col("discharge_time")
            )
            .select(
                "_row_id",
                pl.col("last_event_time").alias("new_time"),
            )
        )

        return (
            df
            .join(
                discharge_updates,
                on="_row_id",
                how="left",
            )
            .with_columns(
                pl.when(pl.col("new_time").is_not_null())
                .then(pl.col("new_time") + pl.duration(seconds=1))
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
    
    @staticmethod
    def resolve_edstay_hadm_id(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """
        Ensure a one-to-one relationship between hadm_id and edstay_id
        where both IDs are present.

        ED stays without a hadm_id are left untouched.
        Hospital stays without an edstay_id are left untouched.

        For ED events with a non-null hadm_id:

        - If a hadm_id has only one edstay_id, it is left unchanged.
        - If a hadm_id has multiple edstay_ids:
            - Keep ED_REGISTRATION events from the temporally first edstay_id.
            - Keep ED_OUT events from the temporally last edstay_id.
            - Remove ED_REGISTRATION events from all other edstay_ids.
            - Remove ED_OUT events from all other edstay_ids.
            - Assign the first edstay_id to all remaining ED events.

        All other rows are preserved.
        """

        required_cols = ["hadm_id", "edstay_id", "time", "code"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        is_ed = pl.col("edstay_id").is_not_null()
        is_registration = pl.col("code").str.starts_with("ED_REGISTRATION")
        is_discharge = pl.col("code").str.starts_with("ED_OUT")

        # Only ED stays with a hadm_id participate in the resolution.
        ed_df = df.filter(
            is_ed & pl.col("hadm_id").is_not_null()
        )

        if ed_df.height == 0:
            return df

        # Find hadm_ids associated with multiple edstay_ids.
        duplicate_hadm_ids = (
            ed_df
            .group_by("hadm_id")
            .agg(
                pl.col("edstay_id").n_unique().alias("_edstay_count")
            )
            .filter(pl.col("_edstay_count") > 1)
            .select("hadm_id")
        )

        if duplicate_hadm_ids.height == 0:
            return df

        # These are guaranteed to be non-null because ed_df was restricted
        # to rows with a non-null hadm_id.
        duplicate_hadm_id_values = duplicate_hadm_ids["hadm_id"].to_list()

        duplicate_ed = ed_df.filter(
            pl.col("hadm_id").is_in(duplicate_hadm_id_values)
        )

        # Determine the temporal start and end of every ED stay.
        stay_times = (
            duplicate_ed
            .group_by(["hadm_id", "edstay_id"])
            .agg(
                pl.col("time").min().alias("_start_time"),
                pl.col("time").max().alias("_end_time"),
            )
        )

        # Temporally first ED stay for each hadm_id.
        first_edstay = (
            stay_times
            .sort(["hadm_id", "_start_time"])
            .group_by("hadm_id", maintain_order=True)
            .first()
            .select([
                "hadm_id",
                pl.col("edstay_id").alias("_first_edstay_id"),
            ])
        )

        # Temporally last ED stay for each hadm_id.
        last_edstay = (
            stay_times
            .sort(["hadm_id", "_end_time"])
            .group_by("hadm_id", maintain_order=True)
            .last()
            .select([
                "hadm_id",
                pl.col("edstay_id").alias("_last_edstay_id"),
            ])
        )

        # Add the first/last stay IDs to the original dataframe.
        #
        # Because first_edstay/last_edstay contain only non-null hadm_ids,
        # rows with hadm_id == null do not participate in the resolution.
        result = (
            df
            .join(
                first_edstay,
                on="hadm_id",
                how="left",
            )
            .join(
                last_edstay,
                on="hadm_id",
                how="left",
            )
        )

        # Remove:
        #   - ED_REGISTRATION events from ED stays other than the first
        #   - ED_OUT events from ED stays other than the last
        #
        # Only duplicated hadm_ids are affected.
        result = result.filter(
            ~(
                is_ed
                & pl.col("hadm_id").is_not_null()
                & pl.col("hadm_id").is_in(duplicate_hadm_id_values)
                & (
                    (
                        is_registration
                        & (
                            pl.col("edstay_id")
                            != pl.col("_first_edstay_id")
                        )
                    )
                    | (
                        is_discharge
                        & (
                            pl.col("edstay_id")
                            != pl.col("_last_edstay_id")
                        )
                    )
                )
            )
        )

        # Give all remaining ED events for the duplicated hadm_id
        # the canonical (first) edstay_id.
        result = (
            result
            .with_columns(
                pl.when(
                    is_ed
                    & pl.col("hadm_id").is_not_null()
                    & pl.col("hadm_id").is_in(duplicate_hadm_id_values)
                )
                .then(pl.col("_first_edstay_id"))
                .otherwise(pl.col("edstay_id"))
                .alias("edstay_id")
            )
            .drop(
                "_first_edstay_id",
                "_last_edstay_id",
            )
        )

        return result

    @staticmethod
    def manual_fixes(df: pl.DataFrame) -> pl.DataFrame:
        time_fixes_adm = {
            23954106: datetime(2163,  4, 21, 1,  0, 0),
            24523858: datetime(2129, 10, 20, 0,  0, 0),
            26728216: datetime(2150,  4, 16, 3, 25, 0),
            27150516: datetime(2144,  4, 30, 0, 52, 0),
            28241666: datetime(2173,  9, 23, 0,  0, 0),
        }
        time_fixes_dc = {
            20327393: datetime(2152, 12, 31,  6,  5, 38),
            24907049: datetime(2129, 11,  2, 13, 50, 21),
        }
        time_fixes_dth = {
            15996626: datetime(2177,  2, 25, 15,  1,  0),
            10535715: datetime(2154, 12,  7,  2, 40, 21),
        }

        fixes_adm_df = pl.DataFrame({
            "hadm_id":  list(time_fixes_adm.keys()),
            "new_time": list(time_fixes_adm.values()),
        })
        fixes_dc_df = pl.DataFrame({
            "hadm_id":  list(time_fixes_dc.keys()),
            "new_time": list(time_fixes_dc.values()),
        })
        fixes_dth_df = pl.DataFrame({
            "subject_id": list(time_fixes_dth.keys()),
            "new_time":   list(time_fixes_dth.values()),
        })

        return (
            df
            # Set the time for the specific target event
            .join(fixes_adm_df, on="hadm_id", how="left")
            .with_columns(
                pl.when(
                    pl.col("new_time").is_not_null()
                    & pl.col("code").str.contains("HOSPITAL_ADMISSION")
                )
                .then(pl.col("new_time"))
                .otherwise(pl.col("time"))
                .alias("time")
            )
            .drop("new_time")
            .join(fixes_dc_df, on="hadm_id", how="left")
            .with_columns(
                pl.when(
                    pl.col("new_time").is_not_null()
                    & (pl.col("code").str.contains("HOSPITAL_DISCHARGE")
                    | pl.col("code").str.contains("DRG//")
                    | pl.col("code").str.contains("DIAGNOSIS//"))
                )
                .then(pl.col("new_time"))
                .otherwise(pl.col("time"))
                .alias("time")
            )
            .drop("new_time")
            .join(fixes_dth_df, on="subject_id", how="left")
            .with_columns(
                pl.when(
                    pl.col("new_time").is_not_null()
                    & pl.col("code").str.contains("MEDS_DEATH")
                )
                .then(pl.col("new_time"))
                .otherwise(pl.col("time"))
                .alias("time")
            )
            .drop("new_time")
            # Temporary key to control ordering of events at the same timestamp
            .with_columns(
                pl.when(pl.col("code").str.contains("HOSPITAL_DISCHARGE"))
                .then(0)
                .when(pl.col("code").str.contains("ED_REGISTRATION"))
                .then(1)
                .when(pl.col("code").str.contains("HOSPITAL_ADMISSION"))
                .then(2)
                .when(pl.col("code").str.contains("ICU_ADMISSION"))
                .then(3)
                .when(pl.col("code").str.contains("ED_OUT"))
                .then(4)
                .otherwise(0)
                .alias("_event_order_1")
            )
            .with_columns(
                pl.when(pl.col("code").str.contains("_DISCHARGE"))
                .then(0)
                .when(pl.col("code").str.contains("DIAGNOSIS//"))
                .then(1)
                .when(pl.col("code").str.contains("DRG//"))
                .then(2)
                .otherwise(0)
                .alias("_event_order_2")
            )
            # Sort discharge before admission when subject_id + time are equal
            .sort(
                [
                    "subject_id",
                    "time",
                    "_event_order_1",
                    "hadm_id",
                    "icustay_id",
                    "_event_order_2"
                ],
                nulls_last=False,
            )
            .drop("_event_order_1", "_event_order_2")
            .with_row_index()
            .drop("index")
        )


class DeathData:
    @staticmethod
    @MatchAndRevise(prefix=[ST.DEATH, ST.DISCHARGE], needs_resorting=True)
    def place_death_before_dc_if_same_time(df: pl.DataFrame) -> pl.DataFrame:
        gb_cols = MatchAndRevise.sort_cols
        idx_col = MatchAndRevise.index_col
        return (
            df.sort(pl.col("code").replace_strict(ST.DEATH, 0, default=1, return_dtype=pl.UInt8))
            .group_by(gb_cols, maintain_order=True)
            .agg(pl.col(idx_col).last(), pl.exclude(gb_cols, idx_col))
            .explode(pl.exclude(gb_cols, idx_col))
            .sort(by=idx_col)
            .select(df.columns)
        )


class DemographicData:
    @staticmethod
    @MatchAndRevise(prefix=ST.ADMISSION)
    def retrieve_demographics_from_hosp_adm(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.concat_list("code", pl.lit("MARITAL_STATUS"), pl.lit("RACE")),
            text_value=pl.concat_list("text_value", pl.col("marital_status"), pl.col("race")),
        ).explode("code", "text_value")

    @staticmethod
    @MatchAndRevise(prefix="RACE", apply_vocab=True)
    def process_race(df: pl.DataFrame) -> pl.DataFrame:
        race_unknown = ["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"]
        race_minor = [
            "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
            "AMERICAN INDIAN/ALASKA NATIVE",
            "MULTIPLE RACE/ETHNICITY",
        ]
        # every patient can have only one race assigned, so we can prioritize which one to keep
        race_priority_mapping = {"RACE//OTHER": 1, "RACE//UNKNOWN": 2}  # every other will get 0
        return (
            df.with_columns(
                code=pl.when(pl.col("text_value").is_in(race_unknown))
                .then(pl.lit("UNKNOWN"))
                .when(pl.col("text_value").is_in(race_minor))
                .then(pl.lit("OTHER"))
                .when(pl.col("text_value") == "SOUTH AMERICAN")
                .then(pl.lit("HISPANIC"))
                .when(pl.col("text_value") == "PORTUGUESE")
                .then(pl.lit("WHITE"))
                .when(pl.col("text_value").str.contains_any(["/", " "]))
                .then(pl.lit(None))
                .otherwise("text_value")
            )
            .with_columns(
                code=(
                    pl.lit("RACE//")
                    + pl.when(pl.col("code").is_null())
                    .then(pl.col("text_value").str.slice(0, pl.col("text_value").str.find("/| ")))
                    .otherwise("code")
                )
            )
            .group_by(MatchAndRevise.sort_cols[0], maintain_order=True)
            .agg(
                pl.col("code")
                .sort_by(
                    pl.col.code.replace_strict(
                        race_priority_mapping, default=0, return_dtype=pl.UInt8
                    )
                )
                .first(),
                pl.exclude("code").first(),
            )
            .select(df.columns)
        )

    @staticmethod
    @MatchAndRevise(prefix="MARITAL_STATUS", apply_vocab=True)
    def process_marital_status(df: pl.DataFrame) -> pl.DataFrame:
        return df.drop_nulls("text_value").with_columns(
            code=pl.lit("MARITAL//") + pl.col("text_value")
        )


class InpatientData:
    @staticmethod
    @MatchAndRevise(prefix="DRG", apply_vocab=True)
    def process_drg_codes(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col.code.str.starts_with("DRG//HCFA")).with_columns(
            code=pl.lit("DRG//") + pl.col.code.str.split("//").list[2].cast(int).cast(str)
        )

    @staticmethod
    @MatchAndRevise(prefix=ST.ADMISSION)
    def process_hospital_admissions(df: pl.DataFrame) -> pl.DataFrame:
        scheduled_admissions = ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]
        return (
            df.with_columns(
                pl.col.code.str.split("//").list[0].alias("code"),
                pl.col.code.str.split("//").list[1].alias("text_value"),
            )
            .with_columns(
                pl.concat_list(
                    "code",
                    pl.lit("ADMISSION_TYPE//")
                    + pl.when(
                        pl.col("text_value").str.ends_with("EMER.")
                        | (pl.col("text_value") == "URGENT")
                    )
                    .then(pl.lit("EMERGENCY"))
                    .when(pl.col("text_value").is_in(scheduled_admissions))
                    .then(pl.lit("SCHEDULED"))
                    .otherwise(pl.lit("OBSERVATION")),
                    pl.lit("INSURANCE//") + pl.col("insurance"),
                ).alias("code")
            )
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix=[ST.DISCHARGE, "DIAGNOSIS//ICD//", "DRG//"])
    def process_hospital_discharges(df: pl.DataFrame) -> pl.DataFrame:
        """Currently must be run before processing diagnoses."""
        discharge_facilities = [
            "HEALTHCARE FACILITY",
            "SKILLED NURSING FACILITY",
            "REHAB",
            "CHRONIC/LONG TERM ACUTE CARE",
            "OTHER FACILITY",
        ]

        drg_following_diag = pl.col.code.str.starts_with(
            "DIAGNOSIS//ICD"
        ) & ~pl.col.code.str.starts_with("DRG//").shift(-1, fill_value=False)
        drg_following_disch = pl.col.code.str.starts_with(ST.DISCHARGE)

        if "stay_id" in df.columns:
            # This means that it is MIMIC with ED extension, and diagnoses in addition come from
            # ED_OUT and in those situations DRG code should not be added
            drg_following_diag &= pl.col.stay_id.is_null() & ~(
                pl.col.code.str.starts_with("DIAGNOSIS//ICD") & pl.col.stay_id.is_null()
            ).shift(-1, fill_value=False)

            drg_following_disch &= pl.col.code.str.starts_with(ST.DISCHARGE).shift(
                -1, fill_value=True
            ) | pl.col.stay_id.is_not_null().shift(-1, fill_value=True)
        else:
            drg_following_diag &= ~pl.col.code.str.starts_with("DIAGNOSIS//ICD").shift(
                -1, fill_value=False
            )
            drg_following_disch &= pl.col.code.str.starts_with(ST.DISCHARGE).shift(
                -1, fill_value=True
            )

        drg_missing_cond = drg_following_diag | drg_following_disch

        return (
            df.with_columns(
                text_value=pl.when(pl.col.code.str.starts_with(ST.DISCHARGE))
                .then(pl.col.code.str.split("//").list[1])
                .otherwise("text_value")
            )
            .with_columns(
                code=pl.when(pl.col.code.str.starts_with(ST.DISCHARGE))
                .then(
                    pl.concat_list(
                        pl.lit(ST.DISCHARGE),
                        (
                            pl.lit("DISCHARGE_LOCATION//")
                            + pl.when(pl.col("text_value").is_in(discharge_facilities))
                            .then(pl.lit("HEALTHCARE_FACILITY"))
                            .when(pl.col("text_value").is_null())
                            .then(pl.lit("UNKNOWN"))
                            .otherwise(pl.col("text_value").replace(" ", "_"))
                        ),
                    )
                )
                .otherwise(pl.concat_list("code")),
                drg_missing=drg_missing_cond,
            )
            .with_columns(
                code=pl.when("drg_missing")
                .then(pl.concat_list("code", pl.lit("DRG//UNKNOWN")))
                .otherwise("code")
            )
            .drop("drg_missing")
            .explode("code")
        )


class MeasurementData:
    @staticmethod
    @MatchAndRevise(prefix=["TEMPERATURE", "HEART_RATE", "RESPIRATORY_RATE", "O2_SATURATION"])
    def process_simple_measurements(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(
                code=pl.concat_list(
                    pl.lit("VITAL//") + pl.col("code"), pl.lit("VITAL//Q//") + pl.col("code")
                )
            )
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix="PAIN")
    def process_pain(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col.text_value.is_not_null())
            .with_columns(
                pl.col("text_value")
                .str.to_lowercase()
                .str.strip_chars(' +"')
                .str.strip_suffix("/10")
            )
            .with_columns(
                numeric_value=pl.when(pl.col.text_value.str.contains("-", literal=True))
                .then(
                    pl.col.text_value.str.split("-").list.first().str.strip_chars() + pl.lit(".5")
                )
                .when(pl.col.text_value.str.contains("crit|moaning"))
                .then(pl.lit("10"))
                .when(pl.col.text_value.str.contains("lot|all over|hurts|much"))
                .then(pl.lit("8"))
                .when(
                    pl.col.text_value.is_in(
                        ["yes", "mild", "moderate", "y", "pain", "uncomfortable"]
                    )
                )
                .then(pl.lit("5"))
                .when(pl.col.text_value.str.contains("little|not bad|some|discomfort"))
                .then(pl.lit("2"))
                .when(
                    pl.col.text_value.str.contains(r"s[lep]{3,4}|sedat|n\/a|resting")
                    | pl.col.text_value.is_in(["no", "no pain", "ok", "none", "comfortable"])
                )
                .then(pl.lit("0"))
                .otherwise(pl.col("text_value").str.replace_all(r"\D", ""))
                .cast(float, strict=False)
            )
            .filter(pl.col.numeric_value.is_between(0, 10))
            .with_columns(code=pl.concat_list(pl.lit("VITAL//PAIN"), pl.lit("VITAL//Q//PAIN")))
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix="Blood Pressure")
    def process_blood_pressure(bp_df: pl.DataFrame) -> pl.DataFrame:
        return (
            bp_df.with_columns(
                code=pl.when(pl.col.numeric_value.is_null()).then(
                    pl.col("text_value").str.split_exact("/", 1)
                )
                # Hacky way to get the systolic and diastolic that come from ED extension
                .otherwise(
                    pl.struct(
                        field_0=pl.col.numeric_value.cast(int).cast(str),
                        field_1="text_value",
                    )
                )
            )
            .with_columns(
                code=pl.concat_list(
                    pl.lit("VITAL//BLOOD_PRESSURE"),
                    pl.lit("VITAL//Q//SBP"),
                    pl.lit("VITAL//Q//DBP"),
                ),
                numeric_value=pl.concat_list(
                    pl.lit(None),
                    pl.col("code").struct[0].cast(float),
                    pl.col("code").struct[1].cast(float),
                ),
                text_value=pl.col.code.struct[0].cast(str)
                + pl.lit("/")
                + pl.col.code.struct[1].cast(str),
            )
            .explode("code", "numeric_value")
        )


class DiagnosesData:
    @staticmethod
    @MatchAndRevise(prefix="DIAGNOSIS//ICD//")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col.code.str.split_exact("//", 3)).with_columns(
            code=pl.lit("ICD//CM//") + pl.col.code.struct[2], text_value=pl.col.code.struct[3]
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//9")
    def convert_icd_9_to_10(icd9_df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_icd_cm_9_to_10_mapping

        icd_9_to_10 = get_icd_cm_9_to_10_mapping()
        return (
            icd9_df.with_columns(
                pl.lit("ICD//CM//10").alias("code"),
                pl.col("text_value").replace_strict(icd_9_to_10, default=None),
            )
        ).drop_nulls("text_value")

    @staticmethod
    @MatchAndRevise(prefix="ICD//CM//10", needs_vocab=True)
    def process_icd10(icd10_df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        from ..mappings import get_icd_cm_code_to_name_mapping

        code_to_name = get_icd_cm_code_to_name_mapping()
        temp_cols = ["part1", "part2", "part3"]
        code_prefixes = ["", "3-6//", "SFX//"]
        code_slices = [(0, 3), (3, 3), (6,)]

        df = (
            icd10_df.with_columns(
                pl.col("text_value").str.slice(*code_slice).alias(col)
                for col, code_slice in zip(temp_cols, code_slices)
            )
            .with_columns(pl.col(temp_cols[0]).replace_strict(code_to_name, default=None))
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(pl.lit(f"ICD//CM//{prefix}") + pl.col(col))
                .alias(col)
                for col, prefix in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )

        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list(temp_cols))
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )


class ProcedureData:
    @staticmethod
    @MatchAndRevise(prefix="PROCEDURE")
    def prepare_codes_for_processing(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col.code.str.split("//"))
            .filter(pl.col.code.list[1] == "ICD")
            .with_columns(
                code=pl.lit("ICD//PCS//") + pl.col.code.list[2], text_value=pl.col.code.list[3]
            )
        )

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//9")
    def convert_icd_9_to_10(icd9_df: pl.DataFrame) -> pl.DataFrame:
        from ..mappings import get_icd_pcs_9_to_10_mapping

        icd_9_to_10 = get_icd_pcs_9_to_10_mapping()
        return (
            icd9_df.with_columns(
                pl.lit("ICD//PCS//10").alias("code"),
                pl.col("text_value").replace(icd_9_to_10, default=None),
            )
        ).drop_nulls("text_value")

    @staticmethod
    @MatchAndRevise(prefix="ICD//PCS//10", needs_vocab=True)
    def process_icd10(icd10_df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        df = icd10_df.with_columns(
            pl.col("text_value").str.split_exact("", 6).alias("code")
        ).with_columns(
            code=pl.concat_list(
                pl.when(pl.col("code").struct[i] != "").then(
                    pl.lit("ICD//PCS//") + pl.col("code").struct[i]
                )
                for i in range(7)
            ).list.drop_nulls()
        )
        if vocab is not None:
            # all characters have to be in the vocab to keep the code
            df = df.filter(pl.col("code").list.eval(pl.element().is_in(vocab)).list.all())
        return df.explode("code").drop_nulls("code")


class MedicationData:
    @staticmethod
    @MatchAndRevise(prefix="MEDICATION", needs_vocab=True)
    def convert_to_atc(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        from ..mappings import get_atc_code_to_desc, get_mimic_drug_name_to_atc_mapping

        drug_to_atc = get_mimic_drug_name_to_atc_mapping()
        code_to_desc = get_atc_code_to_desc()
        temp_cols = ["pfx", "4", "sfx"]
        code_prefixes = ["ATC//", "ATC//4//", "ATC//SFX//"]
        code_slices = [(0, 3), (3, 1), (4,)]

        df = (
            df.with_columns(pl.col("code").str.split("//"))
            .with_columns(
                pl.when(pl.col("code").list[2] == "Administered")
                .then(None)
                .when(pl.col("code").list[1] == "START")
                .then(pl.lit("MEDICATION_START"))
                .alias("code"),
                pl.when(pl.col("code").list[2] == "Administered")
                .then(pl.col("code").list[1])
                .when(pl.col("code").list[1] == "START")
                .then(pl.col("code").list[2])
                .alias("text_value"),
            )
            .drop_nulls("text_value")
            .with_columns(
                pl.col("text_value")
                .str.strip_chars(" ")
                .str.to_lowercase()
                .replace_strict(drug_to_atc, default=None, return_dtype=pl.List(pl.String))
            )
            .with_columns(
                pl.concat_list(
                    "code",
                    pl.lit(None).cast(str).repeat_by(pl.col("text_value").list.len().cast(int) - 1),
                ).alias("code")
            )
            .drop_nulls("text_value")
            .explode("code", "text_value")
            .with_columns(
                pl.col("text_value").str.slice(*slice).alias(col)
                for col, slice in zip(temp_cols, code_slices)
            )
            .with_columns(
                pl.when(pl.col(col) != "")
                .then(
                    pl.lit(pfx)
                    + pl.col(col)
                    + (
                        pl.lit("//") + pl.col(col).replace_strict(code_to_desc, default=None)
                        if pfx == code_prefixes[0]
                        else pl.lit("")
                    )
                )
                .alias(col)
                for col, pfx in zip(temp_cols, code_prefixes)
            )
            .with_columns(unify_code_names(pl.col(temp_cols)))
        )
        if vocab is not None:
            df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

        return (
            df.with_columns(code=pl.concat_list("code", *temp_cols))
            .drop(temp_cols)
            .explode("code")
            .drop_nulls("code")
        )


class ICUStayData:
    @staticmethod
    @MatchAndRevise(prefix="ICU_")
    def process(df: pl.DataFrame, *, num_quantiles: int = 10) -> pl.DataFrame:
        from ..mappings import get_stay_id_to_sofa_mapping

        stay_id_to_sofa = get_stay_id_to_sofa_mapping()
        min_value, max_value = min(stay_id_to_sofa.values()), max(stay_id_to_sofa.values())
        bins = np.linspace(min_value, max_value, num_quantiles + 1)
        values = [
            np.arange(np.ceil(left), np.floor(right) + 1)
            for left, right in zip(bins[:-1], bins[1:])
        ]

        # these are not real quantiles, the values are divided equidistantly
        value_to_quantile = {
            value: f"Q{i}" for i, values in enumerate(values, 1) for value in values
        }

        stay_id_to_sofa = {
            stay_id: value_to_quantile[sofa] for stay_id, sofa in stay_id_to_sofa.items()
        }
        return (
            df.with_columns(pl.col("code").str.split("//"))
            .with_columns(
                code=pl.col("code").list[0],
                text_value=pl.lit("ICU_TYPE//") + pl.col("code").list[1],
                sofa_quantiles=pl.col("icustay_id").replace_strict(stay_id_to_sofa, default=None),
            )
            .with_columns(
                pl.when(code=ST.ICU_ADMISSION)
                .then(
                    pl.concat_list(
                        "code",
                        "text_value",
                        pl.when(pl.col("sofa_quantiles").is_not_null())
                        .then(pl.concat_list(pl.lit(ST.SOFA), "sofa_quantiles"))
                        .otherwise([]),
                    ).alias("code")
                )
                .otherwise(pl.concat_list("code"))
            )
            .drop("sofa_quantiles")
            .explode("code")
        )


class TransferData:
    @staticmethod
    @MatchAndRevise(prefix="TRANSFER_TO", apply_vocab=True)
    def retain_only_transfer_and_admit_types(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col.code.str.split("//"))
            .filter(pl.col.code.list[1].is_in(["transfer", "admit"]))
            .with_columns(code=pl.lit("TRANSFER//") + pl.col.code.list[2].fill_null("UNKNOWN"))
        )


class BMIData:
    @staticmethod
    @MatchAndRevise(prefix="BMI")
    def make_quantiles(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col("text_value").cast(str).cast(float).alias("numeric_value"),
                pl.lit(None).alias("text_value"),
            )
            .filter(pl.col("numeric_value").is_between(10, 100))
            .with_columns(pl.concat_list(pl.lit("BMI"), pl.lit("BMI//Q")).alias("code"))
            .explode("code")
        )

    @staticmethod
    @MatchAndRevise(prefix=["BMI", "Q"])
    def join_token_and_quantile(df: pl.DataFrame) -> pl.DataFrame:
        q_following_bmi_mask = (pl.col("code") == "BMI").shift(1)
        return df.with_columns(
            code=pl.when(q_following_bmi_mask)
            .then(pl.lit("BMI//") + pl.col("code"))
            .when(pl.col("code") == "BMI")
            .then(None)
            .otherwise("code")
        ).drop_nulls("code")


class LabData:
    @staticmethod
    @MatchAndRevise(prefix="LAB//", apply_vocab=True)
    def retain_only_test_with_numeric_result(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("numeric_value").is_not_null())

    @staticmethod
    @MatchAndRevise(prefix="LAB//", needs_counts=True, needs_vocab=True)
    def make_quantiles(
        df: pl.DataFrame, counts: dict[str, int] | None = None, vocab: list[str] | None = None
    ) -> pl.DataFrame:
        # TODO: we've run a simple analysis and decided to keep 200 most frequent labs
        # as the cover most of all the labs in the dataset
        known_lab_names = list(counts.keys())[:200] if vocab is None else vocab
        return (
            df.filter(unify_code_names(pl.col("code")).is_in(known_lab_names))
            .with_columns(pl.concat_list("code", pl.lit("LAB//Q//") + pl.col("code").str.slice(5)))
            .explode("code")
        )


class HCPCSData:
    @staticmethod
    @MatchAndRevise(prefix="HCPCS//", apply_vocab=True)
    def unify_names(df: pl.DataFrame) -> pl.DataFrame:
        """This will just unify the code names."""
        return df


class PatientFluidOutputData:
    @staticmethod
    @MatchAndRevise(prefix="SUBJECT_FLUID_OUTPUT//", needs_vocab=True)
    def make_quantiles(df: pl.DataFrame, vocab: list[str] | None = None) -> pl.DataFrame:
        if vocab is not None:
            df.filter(pl.col("code").is_in(vocab))

        prefix = "SUBJECT_FLUID_OUTPUT//"
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(
                pl.concat_list(
                    "code", pl.lit(prefix + "Q//") + pl.col("code").str.slice(len(prefix))
                )
            )
            .explode("code")
        )


class EdData:
    @staticmethod
    @MatchAndRevise(prefix="ED_REGISTRATION")
    def process_ed_registration(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            code=pl.concat_list(
                "code",
                pl.lit("ED_TRANSPORT//")
                + pl.when(pl.col.text_value == "HELICOPTER")
                .then(pl.lit("OTHER"))
                .otherwise("text_value"),
            )
        ).explode("code")

    @staticmethod
    @MatchAndRevise(prefix="ACUITY")
    def process_ed_acuity(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col("numeric_value").is_not_null())
            .with_columns(
                code=pl.concat_list(
                    "code", pl.lit("Q") + pl.col("numeric_value").cast(pl.UInt8).cast(pl.Utf8)
                )
            )
            .explode("code")
        )
