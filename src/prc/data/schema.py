"""Column names from the challenge data dictionary (data.html).

Movement columns carry the ``_mvt`` postfix; Network Manager flight columns carry
``_flt``. The movement table is left-joined onto flight records, so every ``_flt``
column may be null.
"""

from __future__ import annotations

# --- movement (airport) columns -------------------------------------------------
MVT_ID = "MVT_ID_mvt"
FLIGHT_ID = "FLIGHT_ID_mvt"
FLIGHT_NO = "FLIGHT_mvt"
FLIGHT_RULE = "FLIGHT_RULE_mvt"
ADEP = "ADEP_mvt"
ADES = "ADES_mvt"
PHASE = "PHASE_mvt"  # DEP | ARR
MVT_TIME = "MVT_TIME_UTC_mvt"  # takeoff (DEP) / landing (ARR)
BLOCK_TIME = "BLOCK_TIME_UTC_mvt"  # off-block (DEP) / in-block (ARR); blank for test DEP
SCHED_TIME = "SCHED_TIME_UTC_mvt"
AIRCRAFT_TYPE = "AIRCRAFT_TYPE_mvt"
RUNWAY = "RUNWAY_mvt"
STAND = "STAND_mvt"
TARGET = "TAXITIME_SEC_mvt"  # = MVT_TIME - BLOCK_TIME, seconds, DEP only

# --- Network Manager flight columns ------------------------------------------------
LOBT = "LOBT_flt"
CALLSIGN = "CALLSIGN_flt"
ADEP_FLT = "ADEP_flt"
ADES_FLT = "ADES_flt"
ADES_FILED = "ADES_FILED_flt"
MARKET_SEGMENT = "MARKET_SEGMENT_flt"
IOBT = "IOBT_flt"
FLIGHT_RULE_FLT = "FLIGHT_RULE_flt"
FLIGHT_TYPE = "FLIGHT_TYPE_flt"
AIRCRAFT_TYPE_FLT = "AIRCRAFT_TYPE_flt"
WAKE_CAT = "WK_TBL_CAT_flt"
OPERATOR = "AIRCRAFT_OPERATOR_flt"
EOBT_1 = "EOBT_1_flt"
ARVT_1 = "ARVT_1_flt"
AOBT_3 = "AOBT_3_flt"
ARVT_3 = "ARVT_3_flt"

# --- derived columns we add (never postfixed, so they can't be confused with source) --
AIRPORT = "airport"  # ADEP for DEP rows, ADES for ARR rows
IS_TEST = "is_test"  # row comes from ranking.parquet
SPLIT_MONTH = "month"  # YYYY-MM of MVT_TIME (used for temporal folds)

AIRPORTS: tuple[str, ...] = (
    "EDDF",
    "EDDM",
    "EGLL",
    "EHAM",
    "LEBL",
    "LEMD",
    "LFPG",
    "LIRF",
    "LTFM",
    "LSZH",
)

TIME_COLS: tuple[str, ...] = (
    MVT_TIME,
    BLOCK_TIME,
    SCHED_TIME,
    LOBT,
    IOBT,
    EOBT_1,
    ARVT_1,
    AOBT_3,
    ARVT_3,
)
CAT_COLS: tuple[str, ...] = (
    AIRPORT,
    ADEP,
    ADES,
    PHASE,
    FLIGHT_RULE,
    AIRCRAFT_TYPE,
    RUNWAY,
    STAND,
    MARKET_SEGMENT,
    FLIGHT_RULE_FLT,
    FLIGHT_TYPE,
    AIRCRAFT_TYPE_FLT,
    WAKE_CAT,
    OPERATOR,
    ADES_FILED,
    CALLSIGN,
)
