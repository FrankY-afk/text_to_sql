# Operational & HR Compliance Metric Specifications

This rulebook specifies employee attendance, staffing metrics, and outlet performance.

## Metric: Effective Staff Attendance Rate
- **Business Definition**: Ratio of effective staff days worked to total scheduled staff days in a calendar month.
- **Attendance Code Crediting**:
  - `PR` (Present): 1.0 day credit
  - `HD` (Half Day): 0.5 day credit
  - `WO` (Weekly Off / Rest Day): Scheduled off, not counted in denominator for working day attendance rate
  - `PH` (Public Holiday): Paid holiday, not counted in attendance rate denominator
  - `CL` (Casual Leave): Authorized paid leave (0 day presence)
  - `SL` (Sick Leave): Authorized paid leave (0 day presence)
  - `AB` (Absent): Unplanned absence (0 day presence)
  - `LWP` (Loss of Pay): Unpaid absence (0 day presence)
- **Calculation Formula**:
  ```sql
  SELECT 
    outlet_code,
    (
      (COUNT(CASE WHEN attendance_code = 'PR' THEN 1 END) * 1.0 +
       COUNT(CASE WHEN attendance_code = 'HD' THEN 1 END) * 0.5) * 100.0
    ) / NULLIF(
      COUNT(CASE WHEN attendance_code IN ('PR', 'HD', 'CL', 'SL', 'AB', 'LWP') THEN 1 END),
      0
    ) AS effective_attendance_rate_pct
  FROM staff_attendance
  GROUP BY outlet_code;
  ```
- **Pitfall**: Naive queries often treat `COUNT(DISTINCT staff_no)` without accounting for `HD = 0.5`, or count `WO` and `PH` in the denominator, falsely depressing the attendance rate.

## Metric: Planned vs Unplanned Absence Rate
- **Business Definition**: 
  - **Planned Leave**: `attendance_code IN ('CL', 'SL')`
  - **Unplanned Absence**: `attendance_code IN ('AB', 'LWP')`
- **Calculation Formula**:
  ```sql
  SELECT 
    COUNT(CASE WHEN attendance_code IN ('AB', 'LWP') THEN 1 END) * 100.0 /
    NULLIF(COUNT(CASE WHEN attendance_code IN ('CL', 'SL', 'AB', 'LWP') THEN 1 END), 0) AS unplanned_absence_ratio_pct
  FROM staff_attendance;
  ```

## Metric: Tier-1 Outlet Target Achievement Rate
- **Business Definition**: Percentage of disbursement target achieved by Tier-1 outlets during a target period.
- **Target Comparison Rule**: Compare actual sum of `loan_amount` from `group_loans` with `target_amount` from `outlet_targets`.
- **Calculation Formula**:
  ```sql
  WITH actuals AS (
    SELECT outlet_code, SUM(loan_amount) AS total_disbursed
    FROM group_loans
    WHERE stage IN ('Payout', 'Disbursed')
    GROUP BY outlet_code
  )
  SELECT 
    o.outlet_code,
    o.outlet_name,
    t.target_amount,
    COALESCE(a.total_disbursed, 0.0) AS actual_disbursed,
    (COALESCE(a.total_disbursed, 0.0) * 100.0) / NULLIF(t.target_amount, 0) AS achievement_pct
  FROM outlets o
  JOIN outlet_targets t ON o.outlet_code = t.outlet_code
  LEFT JOIN actuals a ON o.outlet_code = a.outlet_code
  WHERE o.tier = 'T1';
  ```
