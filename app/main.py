"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB   (a local .env file is read, but real environment variables win)

The file is organised top to bottom:
  1. Setup            - settings, database connection, indexes
  2. Time helpers     - epoch-millisecond <-> datetime, IST shift maths (rules R1-R5)
  3. Output helpers   - turn MongoDB documents into the JSON the contract wants
  4. Models           - request bodies with validation
  5. Endpoints
"""
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

# =========================================================================== #
# 1. SETUP
# =========================================================================== #

# Read the .env file if there is one. Real environment variables are NOT overwritten.
load_dotenv()

# tz_aware=True makes PyMongo give back datetimes that know they are UTC
# (the default gives "naive" datetimes with no time zone, which causes silent bugs).
client = MongoClient(
    os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,  # fail fast (5 s) when MongoDB is down instead of hanging
)
db = client[os.environ.get("MONGO_DB", "attendance_db")]


def ensure_indexes() -> None:
    """Create every index we need. Safe to run many times (MongoDB ignores existing ones).

    RULE: never change the definition of a line below once it has run, only add new lines.
    """
    # employees
    db.employees.create_index("emp_code", unique=True, name="uniq_emp_code")  # no duplicate codes
    db.employees.create_index([("department", 1), ("emp_code", 1)], name="dept_emp_code")  # filter + sort
    # attendance_logs
    db.attendance_logs.create_index(  # one record per employee per day, and fast lookup by (emp, date range)
        [("emp_code", 1), ("date", 1)], unique=True, name="uniq_emp_date"
    )
    db.attendance_logs.create_index([("date", -1), ("emp_code", 1)], name="date_desc_emp")  # default list order
    db.attendance_logs.create_index(  # status filter + same list order
        [("status", 1), ("date", -1), ("emp_code", 1)], name="status_date_emp"
    )
    db.attendance_logs.create_index(  # punch-out: newest record with punch_in <= time
        [("emp_code", 1), ("punch_in", -1)], name="emp_punch_in"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_indexes()  # runs once, when the server starts
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(PyMongoError)
async def mongo_error_handler(_request: Request, _exc: PyMongoError):
    """Any unexpected database failure becomes a clean 503 instead of a crash (500)."""
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# =========================================================================== #
# 2. TIME HELPERS  (rules R1 - R5)
# =========================================================================== #

IST = timezone(timedelta(hours=5, minutes=30))  # India time, UTC+05:30
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# An instant in the API: a whole number of milliseconds. The contract's allowed range
# rejects values that look like seconds, and strict=True rejects floats and strings.
EpochMillis = Annotated[int, Field(strict=True, ge=100_000_000_000, le=4_102_444_800_000)]


def ms_to_dt(ms: int) -> datetime:
    """Epoch milliseconds -> UTC datetime, cut down to whole seconds (R1)."""
    return datetime.fromtimestamp(ms // 1000, tz=UTC)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """UTC datetime -> epoch milliseconds (None stays None)."""
    if dt is None:
        return None
    if dt.tzinfo is None:  # safety net: treat a naive datetime as UTC
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)  # integer maths, no float rounding


def now_dt() -> datetime:
    """The current time in UTC, whole seconds."""
    return datetime.now(UTC).replace(microsecond=0)


def hhmm_to_seconds(hhmm: str) -> int:
    """'09:30' -> 34200 (seconds after midnight)."""
    hours, minutes = hhmm.split(":")
    return int(hours) * 3600 + int(minutes) * 60


def is_overnight(shift_start: str, shift_end: str) -> bool:
    """A shift that ends at or before it starts runs past midnight (R1)."""
    return hhmm_to_seconds(shift_end) <= hhmm_to_seconds(shift_start)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """Which IST calendar day does this punch-in belong to? (R1)"""
    local = punch_in.astimezone(IST)
    day = local.date()
    seconds_into_day = local.hour * 3600 + local.minute * 60 + local.second
    # Overnight shift + punch-in earlier than the shift end -> it belongs to yesterday's shift.
    if is_overnight(shift_start, shift_end) and seconds_into_day < hhmm_to_seconds(shift_end):
        day -= timedelta(days=1)
    return day.isoformat()


def shift_bounds(date_str: str, shift_start: str, shift_end: str) -> tuple[datetime, datetime]:
    """The exact start and end moments (IST) of the shift that began on date_str."""
    day = date.fromisoformat(date_str)
    start = datetime.combine(day, datetime.min.time(), tzinfo=IST) + timedelta(seconds=hhmm_to_seconds(shift_start))
    end = datetime.combine(day, datetime.min.time(), tzinfo=IST) + timedelta(seconds=hhmm_to_seconds(shift_end))
    if end <= start:  # overnight: the shift ends on the next calendar day
        end += timedelta(days=1)
    return start, end


def round_half_up(value, places: int) -> float:
    """Round the way people expect (2.675 -> 2.68). Python's round() does not."""
    exact = Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return float(exact)


def compute_late_minutes(punch_in: datetime, shift_start_at: datetime) -> int:
    """R2: late only if MORE than 10:00 after shift start; then whole minutes since shift start."""
    seconds_late = int((punch_in - shift_start_at).total_seconds())
    return seconds_late // 60 if seconds_late > 600 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600, rounded to 2 decimals half-up."""
    seconds = int((punch_out - punch_in).total_seconds())
    return round_half_up(Decimal(seconds) / Decimal(3600), 2)


def compute_overtime(punch_out: datetime, shift_end_at: datetime) -> int:
    """R3: whole minutes after shift end, counted only when at least 30."""
    minutes = int((punch_out - shift_end_at).total_seconds()) // 60
    return minutes if minutes >= 30 else 0


def derive_fields(emp: dict, date_str: str, punch_in: datetime, punch_out: Optional[datetime]) -> dict:
    """Work out late_minutes, work_hours, overtime_minutes and half_day for one record (R2-R5)."""
    start_at, end_at = shift_bounds(date_str, emp["shift_start"], emp["shift_end"])
    fields = {
        "late_minutes": compute_late_minutes(punch_in, start_at),
        "work_hours": None,
        "overtime_minutes": 0,
        "half_day": False,
    }
    if punch_out is not None:  # these three only exist once the employee has punched out
        fields["work_hours"] = compute_work_hours(punch_in, punch_out)
        fields["overtime_minutes"] = compute_overtime(punch_out, end_at)
        fields["half_day"] = fields["work_hours"] < 4.5  # R5 uses the rounded hours
    return fields


# =========================================================================== #
# 3. OUTPUT HELPERS  (what we send back - never _id, instants as milliseconds)
# =========================================================================== #

def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc["created_at"]),
    }


# =========================================================================== #
# 4. MODELS  (request bodies; FastAPI answers 422 automatically when they are invalid)
# =========================================================================== #

HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"  # 24-hour HH:MM  (note [0-9], not \d, which also matches other scripts' digits)


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM)
    shift_end: str = Field(default="18:30", pattern=HHMM)
    joined_on: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")

    @field_validator("joined_on")
    @classmethod
    def joined_on_is_a_real_date(cls, value: str) -> str:
        date.fromisoformat(value)  # raises ValueError for things like 2026-02-31 -> becomes a 422
        return value

    @model_validator(mode="after")
    def shift_times_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


# =========================================================================== #
# 5. ENDPOINTS
# =========================================================================== #

@app.get("/health")
def health():
    """200 only if MongoDB answers a ping, otherwise 503."""
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "MongoDB unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_dt()  # set by the server, never by the client
    try:
        # The UNIQUE INDEX on emp_code is what guarantees no duplicates, even for simultaneous requests.
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query = {} if department is None else {"department": department}
    total = db.employees.count_documents(query)  # counts the FILTERED set (R10)
    cursor = (
        db.employees.find(query, {"_id": 0})
        .sort("emp_code", 1)
        .skip((page - 1) * page_size)  # page 1 skips nothing
        .limit(page_size)
    )
    return {"items": [employee_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}