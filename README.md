# Employee Attendance & Analytics API

A small FastAPI + MongoDB service: punch-in and punch-out with business rules, manual corrections with an audit trail, and analytics built on MongoDB aggregation pipelines.

All the code is in `app/main.py`. The defects I found in the starter are in `REVIEW.md`, and my design answers are in `DECISIONS.md`.

## Run it

Needs Python 3.11+ and MongoDB 6.0 or newer. I developed and tested against MongoDB 7 running in a local Docker container.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Settings come from environment variables (a local `.env` file is also read; real environment variables win):

| Variable    | Meaning                   | Default                     |
| ----------- | ------------------------- | --------------------------- |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGO_DB`  | database name             | `attendance_db`             |

Start the API:

```bash
uvicorn app.main:app --port 8000
```

Indexes are created automatically at startup. Interactive docs: `http://127.0.0.1:8000/docs`.

## Endpoints

| Method and path                                           | What it does                                                                         |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `GET /health`                                             | 200 only if MongoDB answers, otherwise 503                                           |
| `POST /employees`, `GET /employees`                       | create and list employees (validated, paged)                                         |
| `POST /attendance/punch-in`, `POST /attendance/punch-out` | record a day; late minutes, work hours, overtime and half day are computed (R1 - R5) |
| `PATCH /attendance/{emp_code}/{date}`                     | manual correction; recomputes the derived values and appends one history entry       |
| `GET /attendance`                                         | filtered, sorted, paged list                                                         |
| `GET /analytics/employees/{emp_code}/monthly`             | one employee's month                                                                 |
| `GET /analytics/departments/summary`                      | per-department totals for a month                                                    |
| `GET /analytics/leaderboard/late`                         | late-minutes ranking with ties                                                       |
| `GET /analytics/departments/{department}/trend`           | one row per day with a 7-day moving average                                          |
| `GET /admin/explain/{endpoint}`                           | MongoDB `explain` output of an endpoint's main query                                 |

## Notes on how it works

- Instants are stored as UTC datetimes and exchanged as epoch milliseconds. Dates and shift times are IST strings, as in the data model.
- Uniqueness (employee code, one record per employee per day) is enforced by unique indexes, so simultaneous requests cannot create duplicates.
- Analytics read the stored derived values (`work_hours`, `late_minutes`, ...) and never recompute them from the punch times.
- Queries and aggregations are forced onto indexes with hints, and the explain endpoint shows `IXSCAN`.
- The aggregations use `$setWindowFields`, `$densify` and `$unionWith` with `$documents`, which is why MongoDB 6.0+ is required.

## What is missing, and what I am unsure about

- Nothing from the brief is knowingly missing.
- I tested with the sample data plus my own awkward cases (edge times, an overnight shift, a mid-month joiner, ties, simultaneous requests). I did **not** load-test with 100,000 records; the indexes and hints are designed for it, and the explain output shows index scans on my small data.
- `moving_avg_7d` is the mean of the _rounded_ `attendance_rate` values in the window.
- For `department_summary`, the explain endpoint explains the attendance pipeline; the small headcount query on `employees` is separate.
- I used an AI assistant as a tutor and reference while building this, and I went through the code to understand it.
