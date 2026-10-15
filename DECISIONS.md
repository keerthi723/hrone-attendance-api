# DECISIONS.md

1. **Indexes.** Unique `emp_code` (no duplicate employees). Unique `(emp_code, date)`: one record per day, fast per-employee lookups. `(date, emp_code)` and `(status, date, emp_code)` serve the list and month-wide analytics. `(emp_code, punch_in)` finds the record to punch out. Two department indexes serve filters and headcount. Rejected: a plain `emp_code` index on the logs; the unique one already covers it.

2. **Punch-in race.** Neither request checks first; both just insert. The unique `(emp_code, date)` index lets MongoDB accept one, and the other gets a duplicate-key error that I turn into 409. Exactly one 201, one 409.

3. **Ties.** I rank with `$rank`, so equal totals share a rank (1, 2, 2, 4), and `limit` is applied after ranking. Both tied people are returned, even past the limit, ordered by `emp_code`.

4. **Headcount.** It is counted from the employees collection, not the logs, so people with no records still count. Attendance totals come from a separate pipeline and I merge the two; departments without logs get zeros.

5. **At 100x the data**, I would keep daily per-department totals in a small summary collection, updated on each punch, so analytics read a few small documents instead of scanning a month of logs.
