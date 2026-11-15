# Customer Analytics & Retention Rulebook

This rulebook specifies cohort metrics, customer lifecycle calculations, and activity definitions.

## Metric: 30-Day Rolling Churn Rate
- **Business Definition**: Percentage of customers active in baseline 30-day window who performed zero qualifying events in the subsequent 30-day window.
- **Qualifying Event**: An event in `activity_events` with `event_type IN ('login', 'transaction', 'payment', 'transfer')`.
- **Calculation Formula**:
  ```sql
  WITH baseline_cohort AS (
    SELECT DISTINCT user_id
    FROM activity_events
    WHERE event_date >= '2024-01-01' AND event_date < '2024-02-01'
  ),
  retained_users AS (
    SELECT DISTINCT a.user_id
    FROM activity_events a
    JOIN baseline_cohort b ON a.user_id = b.user_id
    WHERE a.event_date >= '2024-02-01' AND a.event_date < '2024-03-01'
  )
  SELECT 
    COUNT(b.user_id) AS baseline_users,
    COUNT(r.user_id) AS retained_users,
    ((COUNT(b.user_id) - COUNT(r.user_id)) * 100.0) / NULLIF(COUNT(b.user_id), 0) AS churn_rate_pct
  FROM baseline_cohort b
  LEFT JOIN retained_users r ON b.user_id = r.user_id;
  ```
- **Boundary Rule**: Use exclusive upper boundaries `< '2024-02-01'` rather than `BETWEEN` to prevent duplicate timestamp double-counting.
