"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB   (a local .env file is read, but real environment variables win)

The file is organised top to bottom:
  1. Setup            - settings, database connection, indexes, error handling
  2. Time helpers     - epoch-millisecond <-> datetime, IST shift maths (rules R1-R5)
  3. Output helpers   - turn MongoDB documents into the JSON the contract wants
  4. Models           - request bodies with validation
  5. Endpoints        - health, employees, punch-in, punch-out, correction (PATCH)
  6. Analytics        - employee monthly, department summary, late leaderboard, department trend
  7. Attendance list and the explain endpoint (proof that every query uses an index)
"""
import calendar
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal, Optional

from bson import Decimal128, json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient, ReturnDocument
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
    db.employees.create_index([("department", 1), ("joined_on", 1)], name="dept_joined")  # headcount queries
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


# =========================================================================== #
# ATTENDANCE: shared types, output helpers and punch-in
# =========================================================================== #

def _check_real_date(value: str) -> str:
    date.fromisoformat(value)  # raises ValueError for 2026-02-31 etc. -> 422
    return value


# A calendar date "YYYY-MM-DD" that must be a real date, and a month "YYYY-MM".
DateStr = Annotated[str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"), AfterValidator(_check_real_date)]
MonthStr = Annotated[str, Field(pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")]

Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]
PRESENCE = ("PRESENT", "WFH", "ON_DUTY")  # statuses that count as "present" (R6)


def history_entry_out(entry: dict) -> dict:
    """One audit-trail entry. Times inside 'changes' are converted to milliseconds too."""
    changes = {}
    for field, change in entry.get("changes", {}).items():
        if field in ("punch_in", "punch_out"):
            changes[field] = {"from": dt_to_ms(change.get("from")), "to": dt_to_ms(change.get("to"))}
        else:
            changes[field] = {"from": change.get("from"), "to": change.get("to")}
    return {"at": dt_to_ms(entry["at"]), "by": entry["by"], "reason": entry["reason"], "changes": changes}


def attendance_out(doc: dict) -> dict:
    """A stored attendance document -> the JSON the contract wants (no _id, old records get defaults)."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,  # missing means 0
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day")),  # missing means false
        "history": [history_entry_out(h) for h in doc.get("history") or []],  # missing means empty
    }


class PunchInIn(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMillis] = None  # milliseconds; leave out to use "now"
    status: PresenceStatus = "PRESENT"  # ABSENT / LEAVE are not allowed here


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")  # the starter crashed with a 500 here

    punched_at = ms_to_dt(body.punched_at) if body.punched_at is not None else now_dt()
    day = attendance_date(punched_at, emp["shift_start"], emp["shift_end"])  # R1
    doc = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status,
        "punch_in": punched_at,
        "punch_out": None,
        **derive_fields(emp, day, punched_at, None),  # late_minutes (R2); the rest start empty
        "history": [],
    }
    try:
        # No "check first, then insert": the UNIQUE index on (emp_code, date) decides the winner,
        # so of two simultaneous requests exactly one succeeds and the other gets 409.
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return attendance_out(doc)


# =========================================================================== #
# ATTENDANCE: punch-out and manual correction (regularization)
# =========================================================================== #

MAX_SHIFT = timedelta(hours=24)  # a punch-out may be at most 24 hours after its punch-in


class PunchOutIn(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMillis] = None  # milliseconds; leave out to use "now"


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    out_at = ms_to_dt(body.punched_at) if body.punched_at is not None else now_dt()

    # The newest record whose punch-in is at or before this punch-out. Looking at the punch-in TIME
    # (not today's date) is what makes overnight shifts work across midnight.
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": out_at}},
        sort=[("punch_in", -1)],
    )
    if record is None:
        raise HTTPException(404, "no punch-in found")
    if record.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")

    in_at = record["punch_in"]
    if out_at <= in_at or out_at - in_at > MAX_SHIFT:
        raise HTTPException(422, "punched_at must be after the punch-in and within 24 hours of it")

    derived = derive_fields(emp, record["date"], in_at, out_at)  # R3, R4, R5
    # Atomic "close it only if it is still open": of two simultaneous punch-outs, exactly one matches.
    # late_minutes is not touched: it was fixed at punch-in.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},  # None matches "null or missing"
        {"$set": {
            "punch_out": out_at,
            "work_hours": derived["work_hours"],
            "overtime_minutes": derived["overtime_minutes"],
            "half_day": derived["half_day"],
        }},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:  # somebody else closed it between our read and our write
        raise HTTPException(409, "already punched out")
    return attendance_out(updated)


class RegularizeIn(BaseModel):
    # Only these can be changed. work_hours, late_minutes etc. are not accepted (extra fields are ignored).
    status: Optional[Status] = None
    punch_in: Optional[EpochMillis] = None
    punch_out: Optional[EpochMillis] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


TRACKED_FIELDS = ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day")


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: DateStr, body: RegularizeIn):  # "date" is the record's day, YYYY-MM-DD
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if record is None:
        raise HTTPException(404, "no attendance record for that date")

    # 1) What the record looks like now (missing fields get their defaults).
    old = {
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes") or 0,
        "overtime_minutes": record.get("overtime_minutes") or 0,
        "half_day": bool(record.get("half_day")),
    }

    # 2) What it should look like after the correction ("omitted fields are unchanged").
    new_status = body.status or old["status"]
    if new_status in PRESENCE:
        new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else old["punch_in"]
        new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else old["punch_out"]
        if new_in is None:
            raise HTTPException(422, "a presence status needs a punch_in")
        if attendance_date(new_in, emp["shift_start"], emp["shift_end"]) != date:  # R1
            raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if new_out is not None and (new_out <= new_in or new_out - new_in > MAX_SHIFT):
            raise HTTPException(422, "punch_out must be after punch_in and within 24 hours of it")
        new = {"status": new_status, "punch_in": new_in, "punch_out": new_out,
               **derive_fields(emp, date, new_in, new_out)}  # recompute R2-R5 from the final times
    else:  # ABSENT or LEAVE: no punch times at all
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "ABSENT and LEAVE cannot have punch times")
        new = {"status": new_status, "punch_in": None, "punch_out": None, "work_hours": None,
               "late_minutes": 0, "overtime_minutes": 0, "half_day": False}

    # 3) A request that changes nothing is rejected.
    if all(new[f] == old[f] for f in ("status", "punch_in", "punch_out")):
        raise HTTPException(422, "the request changes nothing")

    # 4) The audit entry lists only the fields that really changed.
    changes = {f: {"from": old[f], "to": new[f]} for f in TRACKED_FIELDS if old[f] != new[f]}
    entry = {"at": now_dt(), "by": body.regularized_by, "reason": body.reason, "changes": changes}

    # 5) Write it only if the record is still exactly as we read it (same status, times and history length).
    #    If another correction or a punch-out slipped in meanwhile, nothing is written and we answer 409,
    #    so no history entry is ever silently lost.
    guard = {"_id": record["_id"], "status": old["status"],
             "punch_in": record.get("punch_in"), "punch_out": record.get("punch_out")}
    history = record.get("history")
    if history is None:  # legacy record without a history field
        guard["history"] = None
        update = {"$set": {**new, "history": [entry]}}
    else:
        guard["history"] = {"$size": len(history)}
        update = {"$set": new, "$push": {"history": entry}}  # $push APPENDS, it never overwrites
    updated = db.attendance_logs.find_one_and_update(guard, update, return_document=ReturnDocument.AFTER)
    if updated is None:
        raise HTTPException(409, "the record was changed by someone else, please retry")
    return attendance_out(updated)


# =========================================================================== #
# ANALYTICS: employee monthly and department summary
# =========================================================================== #

# Index hints. Without a hint MongoDB may choose a full scan when most documents match;
# the hint forces the index, so the explain endpoint can always prove "IXSCAN, no COLLSCAN".
HINT_EMP_DATE = "uniq_emp_date"  # attendance_logs: (emp_code, date)
HINT_DATE_EMP = "date_desc_emp"  # attendance_logs: (date, emp_code)
HINT_DEPT_JOINED = "dept_joined"  # employees: (department, joined_on)


def month_bounds(month: str) -> tuple[date, date]:
    """'2026-07' -> (first day, last day) of that month."""
    year, mon = int(month[:4]), int(month[5:7])
    if year < 1:
        raise HTTPException(422, "invalid month")
    return date(year, mon, 1), date(year, mon, calendar.monthrange(year, mon)[1])


def weekdays_between(first: date, last: date) -> int:
    """How many Mon-Fri days from first to last, both included (R7). 0 if first is after last."""
    return sum(1 for i in range((last - first).days + 1) if (first + timedelta(days=i)).weekday() < 5)


def weekday_dates(first: date, last: date) -> list[str]:
    """Every Monday-Friday date in the range, as 'YYYY-MM-DD' strings."""
    days = (first + timedelta(days=i) for i in range((last - first).days + 1))
    return [d.isoformat() for d in days if d.weekday() < 5]


def to_decimal(value) -> Decimal:
    """MongoDB may hand back Decimal128, a float or an int - turn any of them into a Python Decimal."""
    if isinstance(value, Decimal128):
        return value.to_decimal()
    return Decimal(str(value))


# ---- small pieces of MongoDB aggregation expressions, reused by several pipelines ---- #

IS_PRESENT = {"$in": ["$status", list(PRESENCE)]}  # R6
LATE_MINUTES = {"$ifNull": ["$late_minutes", 0]}  # missing means 0


def present_days_expr(weekdays: list[str]) -> dict:
    """Per record: 1 for a present weekday, 0.5 for a present half-day weekday, else 0 (R5-R7)."""
    return {"$cond": [
        {"$and": [IS_PRESENT, {"$in": ["$date", weekdays]}]},
        {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]},
        0,
    ]}


def count_if(condition: dict) -> dict:
    """$sum this to count the records where condition is true."""
    return {"$sum": {"$cond": [condition, 1, 0]}}


# ---------------------------------------------------------------------------- #
# Employee monthly
# ---------------------------------------------------------------------------- #

def employee_monthly_pipeline(emp_code: str, month: str) -> list:
    first, last = month_bounds(month)
    return [
        # Uses the (emp_code, date) index: one employee, one date range.
        {"$match": {"emp_code": emp_code, "date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {"$group": {
            "_id": None,
            "present_days": {"$sum": present_days_expr(weekday_dates(first, last))},
            "leave_days": count_if({"$eq": ["$status", "LEAVE"]}),
            "late_count": count_if({"$gt": [LATE_MINUTES, 0]}),
            "total_late_minutes": {"$sum": LATE_MINUTES},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: MonthStr):
    emp = db.employees.find_one({"emp_code": emp_code}, {"joined_on": 1})
    if emp is None:
        raise HTTPException(404, "employee not found")

    first, last = month_bounds(month)
    # R7: working days (Mon-Fri) are counted from the joining date when it falls inside the month.
    working_days = weekdays_between(max(first, date.fromisoformat(emp["joined_on"])), last)

    rows = list(db.attendance_logs.aggregate(employee_monthly_pipeline(emp_code, month), hint=HINT_EMP_DATE))
    stats = rows[0] if rows else {}  # no records at all -> every number is 0
    present_days = float(stats.get("present_days", 0))
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": round_half_up(present_days, 2),
        "leave_days": int(stats.get("leave_days", 0)),
        "late_count": int(stats.get("late_count", 0)),
        "total_late_minutes": int(stats.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(stats.get("total_overtime_minutes", 0)),
        "attendance_pct": None if working_days == 0
        else round_half_up(Decimal(str(present_days)) * 100 / working_days, 2),
    }


# ---------------------------------------------------------------------------- #
# Department summary
# ---------------------------------------------------------------------------- #

def department_headcount_pipeline(month: str, department: Optional[str]) -> list:
    """R9: employees who joined on or before the month's last day - counted from the EMPLOYEES collection,
    so people with zero attendance records are included."""
    _first, last = month_bounds(month)
    match = {"joined_on": {"$lte": last.isoformat()}}
    if department is not None:
        match["department"] = department
    return [{"$match": match}, {"$group": {"_id": "$department", "n": {"$sum": 1}}}]


def department_summary_pipeline(month: str, department: Optional[str]) -> list:
    """Attendance numbers per department. The department lives on the employee, so we join to it."""
    first, last = month_bounds(month)
    emp_filter = {"emp.joined_on": {"$lte": last.isoformat()}}  # people who have not joined yet are left out
    if department is not None:
        emp_filter["emp.department"] = department
    has_hours = {"$and": [IS_PRESENT, {"$gt": ["$work_hours", None]}]}  # present AND a real work_hours value
    return [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},  # (date, emp_code) index
        {"$lookup": {"from": "employees", "localField": "emp_code", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # a log whose employee does not exist disappears here
        {"$match": emp_filter},
        {"$group": {
            "_id": "$emp.department",
            "present_days": {"$sum": present_days_expr(weekday_dates(first, last))},
            # Average over RECORDS: keep the exact sum and the count, divide once at the end.
            "wh_sum": {"$sum": {"$cond": [has_hours, {"$toDecimal": "$work_hours"}, 0]}},
            "wh_count": count_if(has_hours),
            "late_count": count_if({"$gt": [LATE_MINUTES, 0]}),
            "total_late_minutes": {"$sum": LATE_MINUTES},
            "leave_count": count_if({"$eq": ["$status", "LEAVE"]}),
            "on_duty_count": count_if({"$eq": ["$status", "ON_DUTY"]}),
        }},
    ]


@app.get("/analytics/departments/summary")
def department_summary(month: MonthStr, department: Optional[str] = None):
    headcounts = {
        row["_id"]: row["n"]
        for row in db.employees.aggregate(department_headcount_pipeline(month, department), hint=HINT_DEPT_JOINED)
    }
    stats = {
        row["_id"]: row
        for row in db.attendance_logs.aggregate(department_summary_pipeline(month, department), hint=HINT_DATE_EMP)
    }

    items = []
    for name in sorted(headcounts):  # departments with headcount 0 never appear; sorted A-Z
        s = stats.get(name, {})  # a department with people but no logs -> all zeros
        hours_count = int(s.get("wh_count", 0))
        items.append({
            "department": name,
            "headcount": headcounts[name],
            "present_days": round_half_up(float(s.get("present_days", 0)), 2),
            "avg_work_hours": round_half_up(to_decimal(s["wh_sum"]) / hours_count, 2) if hours_count else None,
            "late_count": int(s.get("late_count", 0)),
            "total_late_minutes": int(s.get("total_late_minutes", 0)),
            "leave_count": int(s.get("leave_count", 0)),
            "on_duty_count": int(s.get("on_duty_count", 0)),
        })
    return {"month": month, "items": items}


# =========================================================================== #
# ANALYTICS: late leaderboard and department trend
# =========================================================================== #

HINT_DEPT_EMP = "dept_emp_code"  # employees: (department, emp_code)
WINDOW_7_ROWS = {"documents": [-6, "current"]}  # this row and up to 6 rows before it

# One record's worth of "presence": 1, or 0.5 for a half day, or 0 when not present (any day of the week).
PRESENT_VALUE = {"$cond": [IS_PRESENT, {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]}, 0]}


# ---------------------------------------------------------------------------- #
# Late leaderboard
# ---------------------------------------------------------------------------- #

def late_leaderboard_pipeline(month: str, limit: int, department: Optional[str]) -> list:
    first, last = month_bounds(month)
    employee_filter = [] if department is None else [{"$match": {"emp.department": department}}]
    return [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},  # (date, emp_code) index
        {"$group": {  # one row per employee
            "_id": "$emp_code",
            "total_late_minutes": {"$sum": LATE_MINUTES},
            "late_count": count_if({"$gt": [LATE_MINUTES, 0]}),
        }},
        {"$match": {"total_late_minutes": {"$gt": 0}}},  # only people who were late at all
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # logs of unknown employees are ignored
        *employee_filter,  # ranking happens INSIDE the department when one is given
        # RANK, not array position: ties share a rank and the next rank is skipped (1, 2, 2, 4).
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit is applied AFTER ranking, so ties can exceed it
        {"$sort": {"total_late_minutes": -1, "_id": 1}},  # highest first, then emp_code A-Z
        {"$project": {
            "_id": 0, "rank": 1, "emp_code": "$_id", "name": "$emp.name", "department": "$emp.department",
            "total_late_minutes": 1, "late_count": 1,
        }},
    ]


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: MonthStr,
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    rows = db.attendance_logs.aggregate(late_leaderboard_pipeline(month, limit, department), hint=HINT_DATE_EMP)
    items = [
        {
            "rank": int(row["rank"]),
            "emp_code": row["emp_code"],
            "name": row["name"],
            "department": row["department"],
            "total_late_minutes": int(row["total_late_minutes"]),
            "late_count": int(row["late_count"]),
        }
        for row in rows
    ]
    return {"month": month, "items": items}


# ---------------------------------------------------------------------------- #
# Department trend
# ---------------------------------------------------------------------------- #

def build_department_trend_pipeline(department: str, date_from: str, date_to: str) -> list:
    """Checks the inputs (422 / 404) and builds the pipeline. Shared by the endpoint and the explain endpoint."""
    first, last = date.fromisoformat(date_from), date.fromisoformat(date_to)
    if last < first:
        raise HTTPException(422, "'to' must not be before 'from'")
    days = (last - first).days + 1
    if days > 92:
        raise HTTPException(422, "the range must not exceed 92 days")

    # The employees of this department (index-only query). No employees at all -> unknown department.
    codes = [
        e["emp_code"]
        for e in db.employees.find({"department": department}, {"_id": 0, "emp_code": 1}).hint(HINT_DEPT_EMP)
    ]
    if not codes:
        raise HTTPException(404, "department not found")

    range_start = datetime.combine(first, datetime.min.time(), tzinfo=UTC)
    range_stop = range_start + timedelta(days=days)  # $densify's upper bound is exclusive
    rate_applies = {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]}  # R: working day AND headcount > 0

    return [
        # 1. This department's records in the range (uses the (date, emp_code) index).
        {"$match": {"date": {"$gte": date_from, "$lte": date_to}, "emp_code": {"$in": codes}}},
        # 2. One row per day that HAS records.
        {"$group": {
            "_id": "$date",
            "present_count": {"$sum": PRESENT_VALUE},
            "late_count": count_if({"$gt": [LATE_MINUTES, 0]}),
        }},
        # 3. A zero row for the first day, so the gap-filling below still works when there are no records at all.
        {"$unionWith": {"pipeline": [{"$documents": [{"_id": date_from, "present_count": 0, "late_count": 0}]}]}},
        {"$group": {"_id": "$_id", "present_count": {"$sum": "$present_count"}, "late_count": {"$sum": "$late_count"}}},
        # 4. GAP FILLING: $densify adds a row for every missing calendar day between from and to.
        {"$addFields": {"day": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}}}},
        {"$densify": {"field": "day", "range": {"step": 1, "unit": "day", "bounds": [range_start, range_stop]}}},
        {"$addFields": {
            "date": {"$dateToString": {"date": "$day", "format": "%Y-%m-%d"}},
            "present_count": {"$ifNull": ["$present_count", 0]},  # the new rows have no counts yet -> 0
            "late_count": {"$ifNull": ["$late_count", 0]},
            "weekday": {"$dayOfWeek": "$day"},  # 1 = Sunday ... 7 = Saturday
        }},
        {"$addFields": {"is_working_day": {"$and": [{"$gte": ["$weekday", 2]}, {"$lte": ["$weekday", 6]}]}}},
        # 5. Headcount for each day: employees of the department who had joined by that day (R9).
        {"$lookup": {
            "from": "employees",
            "let": {"day_text": "$date"},
            "pipeline": [
                {"$match": {"department": department}},
                {"$match": {"$expr": {"$lte": ["$joined_on", "$$day_text"]}}},
                {"$count": "n"},
            ],
            "as": "joined",
        }},
        {"$addFields": {"headcount": {"$ifNull": [{"$first": "$joined.n"}, 0]}}},
        # 6. attendance_rate in ten-thousandths, rounded HALF-UP with whole-number maths:
        #    floor((present * 20000 + headcount) / (2 * headcount))  ==  round_half_up(present / headcount * 10000)
        {"$addFields": {
            "rate_x10000": {"$cond": [rate_applies, {"$floor": {"$divide": [
                {"$add": [{"$multiply": ["$present_count", 20000]}, "$headcount"]},
                {"$multiply": [2, "$headcount"]},
            ]}}, None]},
            "has_rate": {"$cond": [rate_applies, 1, 0]},
        }},
        # 7. MOVING AVERAGE: add up the rates (and count them) over this row and the 6 rows before it.
        {"$setWindowFields": {"sortBy": {"day": 1}, "output": {
            "window_sum": {"$sum": "$rate_x10000", "window": WINDOW_7_ROWS},
            "window_count": {"$sum": "$has_rate", "window": WINDOW_7_ROWS},
        }}},
        {"$sort": {"day": 1}},
        {"$project": {
            "_id": 0, "date": 1, "is_working_day": 1, "headcount": 1, "present_count": 1, "late_count": 1,
            "rate_x10000": 1, "window_sum": 1, "window_count": 1,
        }},
    ]


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: Annotated[DateStr, Query(alias="from")],
    date_to: Annotated[DateStr, Query(alias="to")],
):
    pipeline = build_department_trend_pipeline(department, date_from, date_to)
    items = []
    for row in db.attendance_logs.aggregate(pipeline, hint=HINT_DATE_EMP):
        rate = row.get("rate_x10000")  # None on weekends and when headcount is 0
        window_count = int(row["window_count"])
        items.append({
            "date": row["date"],
            "is_working_day": bool(row["is_working_day"]),
            "headcount": int(row["headcount"]),
            "present_count": round_half_up(row["present_count"], 2),
            "late_count": int(row["late_count"]),
            "attendance_rate": None if rate is None else float(Decimal(int(rate)) / 10000),
            # mean of the (rounded) rates in the window, 4 decimals half-up; None if the window has no rate
            "moving_avg_7d": None if window_count == 0
            else round_half_up(Decimal(int(row["window_sum"])) / Decimal(window_count) / 10000, 4),
        })
    return {"department": department, "items": items}


# =========================================================================== #
# ATTENDANCE LIST (filter, sort, page) and the EXPLAIN endpoint
# =========================================================================== #

ATTENDANCE_SORT = [("date", -1), ("emp_code", 1)]  # newest day first, then by employee code


def build_attendance_query(
    emp_code: Optional[str], date_from: Optional[str], date_to: Optional[str], status: Optional[str]
) -> dict:
    """The MongoDB filter for the attendance list. Shared by the list endpoint and the explain endpoint."""
    if date_from and date_to and date_from > date_to:  # ISO dates compare correctly as text
        raise HTTPException(422, "date_from must not be after date_to")
    query: dict = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    date_range = {}
    if date_from:
        date_range["$gte"] = date_from
    if date_to:
        date_range["$lte"] = date_to
    if date_range:
        query["date"] = date_range
    if status:
        query["status"] = status
    return query


def attendance_hint(query: dict) -> str:
    """Pick the index that fits the filter, so MongoDB never reads the whole collection."""
    if "emp_code" in query:
        return HINT_EMP_DATE  # (emp_code, date)
    if "status" in query:
        return "status_date_emp"  # (status, date desc, emp_code)
    return HINT_DATE_EMP  # (date desc, emp_code)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query = build_attendance_query(emp_code, date_from, date_to, status)
    hint = attendance_hint(query)
    total = db.attendance_logs.count_documents(query, hint=hint)
    cursor = (
        db.attendance_logs.find(query, {"_id": 0})
        .hint(hint)
        .sort(ATTENDANCE_SORT)  # the DATABASE sorts and slices; we never load everything
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {"items": [attendance_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


# ---------------------------------------------------------------------------- #
# Explain endpoint
# ---------------------------------------------------------------------------- #

ExplainTarget = Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"]


def run_explain(command: dict) -> dict:
    """Ask MongoDB to explain a command (with real execution statistics) and return it as plain JSON."""
    result = db.command({"explain": command, "verbosity": "executionStats"})
    return json.loads(json_util.dumps(result))  # converts dates, ObjectIds, etc. into JSON-safe values


def explain_aggregate(collection: str, pipeline: list, hint: str) -> dict:
    return run_explain({"aggregate": collection, "pipeline": pipeline, "cursor": {}, "hint": hint})


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: ExplainTarget,
    emp_code: Optional[str] = None,
    month: Optional[MonthStr] = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[Status] = None,
    from_date: Annotated[Optional[DateStr], Query(alias="from")] = None,
    to_date: Annotated[Optional[DateStr], Query(alias="to")] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Explains the main query of one endpoint, built by the SAME helper functions that endpoint uses."""
    collection = "attendance_logs"

    if endpoint == "attendance_list":
        query = build_attendance_query(emp_code, date_from, date_to, status)
        explain = run_explain({
            "find": collection,
            "filter": query,
            "sort": dict(ATTENDANCE_SORT),
            "skip": (page - 1) * page_size,
            "limit": page_size,
            "projection": {"_id": 0},
            "hint": attendance_hint(query),
        })
    elif endpoint == "employee_monthly":
        if emp_code is None or month is None:
            raise HTTPException(422, "emp_code and month are required")
        explain = explain_aggregate(collection, employee_monthly_pipeline(emp_code, month), HINT_EMP_DATE)
    elif endpoint == "department_summary":
        if month is None:
            raise HTTPException(422, "month is required")
        explain = explain_aggregate(collection, department_summary_pipeline(month, department), HINT_DATE_EMP)
    elif endpoint == "late_leaderboard":
        if month is None:
            raise HTTPException(422, "month is required")
        explain = explain_aggregate(collection, late_leaderboard_pipeline(month, limit, department), HINT_DATE_EMP)
    else:  # department_trend
        if department is None or from_date is None or to_date is None:
            raise HTTPException(422, "department, from and to are required")
        explain = explain_aggregate(
            collection, build_department_trend_pipeline(department, from_date, to_date), HINT_DATE_EMP
        )
    return {"endpoint": endpoint, "collection": collection, "explain": explain}
