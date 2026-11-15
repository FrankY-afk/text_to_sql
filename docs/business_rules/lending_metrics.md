# Lending Domain Business Rules & Metric Specifications

This rulebook defines standardized financial and credit metrics for lending operations.

## Metric: Non-Performing Assets (NPA / 90+ DPD)
- **Business Definition**: A loan account is classified as a Non-Performing Asset (NPA) or severely delinquent if Days Past Due (DPD) is 90 days or greater.
- **Tables Required**: `overdue_daily`, `group_loans` or `individual_loans`
- **Columns Required**: `dpd`, `loan_id`, `overdue_amount`
- **Calculation Formula**:
  ```sql
  -- Loans with DPD >= 90 as of evaluation date
  SELECT loan_id, MAX(dpd) AS max_dpd, SUM(overdue_amount) AS total_npa_overdue
  FROM overdue_daily
  WHERE dpd >= 90
  GROUP BY loan_id;
  ```
- **Filter Rule**: Only include accounts with `dpd >= 90`. Accounts with DPD between 1 and 89 are categorized as SMA (Special Mention Accounts: SMA-0 for 1-30, SMA-1 for 31-60, SMA-2 for 61-89).
- **Pitfall**: Do not include accounts with `dpd < 90` in NPA calculations. Always aggregate daily snapshots by `loan_id` before joining to parent loan tables to avoid join fan-out.

## Metric: Active Loan Portfolio (Gross Loan Portfolio / GLP)
- **Business Definition**: The total outstanding principal or volume of active loans that have been disbursed and not yet closed or written off.
- **Tables Required**: `group_loans` or `individual_loans`
- **Columns Required**: `stage`, `loan_amount`, `loan_id`
- **Calculation Formula**:
  ```sql
  -- Active loans are in stage 'Payout' or 'Disbursed', excluding 'Closed', 'Rejected', or 'Cancelled'
  SELECT SUM(loan_amount) AS gross_loan_portfolio, COUNT(DISTINCT loan_id) AS active_loan_count
  FROM group_loans
  WHERE stage IN ('Payout', 'Disbursed', 'Active');
  ```
- **Filter Rule**: Exclude `stage IN ('Rejected', 'Closed', 'Cancelled', 'Enrolment', 'Ready')`.

## Metric: Net Disbursed Payout (Net Volume)
- **Business Definition**: Net disbursement amount credited to the borrower after deducting upfront processing fees, insurance charges, and stamp duties.
- **Processing Fee Rule**: Upfront deduction is 2.5% of `loan_amount` or fixed fee of 500 currency units, whichever is specified in the contract.
- **Calculation Formula**:
  ```sql
  SELECT 
    loan_id, 
    loan_amount, 
    (loan_amount * 0.975) AS net_payout_amount
  FROM individual_loans
  WHERE stage IN ('Payout', 'Disbursed');
  ```

## Metric: Overdue Loan Provisioning Coverage
- **Business Definition**: Reserve capital required to cover anticipated credit losses based on delinquency bucket:
  - Standard (0 DPD): 0.4% provision
  - SMA-1 & SMA-2 (1-89 DPD): 5% provision
  - Substandard / NPA (90-180 DPD): 25% provision
  - Doubtful (>180 DPD): 100% provision
