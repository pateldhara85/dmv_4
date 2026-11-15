"""
Northbridge Bank - Credit Risk Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline (unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once, generated route only) -> execute_query
    -> generate_response, with escalation to a human analyst if validation fails.

Run locally:
    streamlit run app.py
"""

import json
import os
import re
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401  (kept from the notebook imports)

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Northbridge Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

# Model names are the same as the notebook; override with env vars/secrets if needed.
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", "gpt-4o")

# =============================================================================
# CONFIGURATION: credentials, database, ground-truth CSV
# =============================================================================


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Look up a setting in st.secrets, then environment variables, then config.json."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    value = os.environ.get(name)
    if value:
        return value

    for cfg_path in (APP_DIR / "config.json", Path("config.json")):
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as file:
                    data = json.load(file)
                if data.get(name):
                    return data[name]
            except Exception:
                pass
    return default


def configure_credentials():
    """Equivalent of the notebook's config.json loading: store credentials as env vars."""
    api_key = get_setting("OPENAI_API_KEY")
    api_base = get_setting("OPENAI_API_BASE") or get_setting("OPENAI_BASE_URL")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
        os.environ["OPENAI_API_BASE"] = api_base
    return api_key, api_base


@st.cache_resource(show_spinner=False)
def get_llms(api_key: str, api_base: Optional[str]):
    """LLM set-up (notebook: llm = ChatOpenAI(gpt-4o-mini), evaluator_llm = ChatOpenAI(gpt-4o))."""
    extra = {"base_url": api_base} if api_base else {}
    llm_ = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=api_key, **extra)
    evaluator_llm_ = ChatOpenAI(model=EVALUATOR_MODEL, temperature=0, api_key=api_key, **extra)
    return llm_, evaluator_llm_


def resolve_db_path() -> Optional[Path]:
    """Find the SQLite file. Tries DB_PATH, then the file names used in the notebook."""
    candidates = [
        get_setting("DB_PATH"),
        "credit_risk_portfolio.db",
        "credit_risk_portfolio(3).db",
        "credit_risk_portfolio (3).db",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        for path in (Path(candidate), APP_DIR / candidate):
            if path.exists():
                return path.resolve()
    return None


@st.cache_resource(show_spinner=False)
def get_connection(db_path_str: str):
    """Read-only SQLite connection (URI mode with ?mode=ro) - no write operations possible."""
    uri = Path(db_path_str).resolve().as_uri() + "?mode=ro"
    # check_same_thread=False because Streamlit may run reruns on different threads.
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def load_ground_truth() -> Optional[pd.DataFrame]:
    for candidate in (APP_DIR / "test_queries.csv", Path("test_queries.csv")):
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


# =============================================================================
# DATABASE SCHEMA (provided to the LLM in the prompts)
# =============================================================================

database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# =============================================================================
# VERIFIED QUERY TEMPLATE LIBRARY
# =============================================================================

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """SELECT s.sector_name,
       ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS outstanding_millions,
       ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard','Doubtful','Loss')
                      THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_millions,
       COUNT(*) AS loan_count,
       SUM(CASE WHEN l.asset_classification IN ('Substandard','Doubtful','Loss') THEN 1 ELSE 0 END) AS npa_loan_count
FROM loan_master AS l
JOIN sector_master AS s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY outstanding_millions DESC"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """SELECT loan_category,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS outstanding_millions
FROM loan_master
GROUP BY loan_category
ORDER BY outstanding_millions DESC"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """ SELECT ifrs9_stage,
       COUNT(*) AS loan_count,
       ROUND(SUM(ead_amount) / 1000000.0, 2) AS ead_millions,
       ROUND(SUM(ecl_amount) / 1000000.0, 2) AS ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """SELECT s.sector_name,
       ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning AS p
JOIN loan_master AS l ON p.loan_account_number = l.loan_account_number
JOIN sector_master AS s ON l.sector_code = s.sector_code
WHERE p.reporting_date = (SELECT MAX(reporting_date) FROM provisioning)
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC"""

# VQ5: Top 10 Loan Exposures
sql_5 = """ SELECT loan_account_number, borrower_name, sector_name,
       ROUND(total_outstanding / 1000000.0, 2) AS outstanding_millions,
       asset_classification
FROM loan_master AS l
JOIN sector_master AS s ON l.sector_code = s.sector_code
ORDER BY total_outstanding DESC
LIMIT 10"""

# VQ6: Top 5 Business Group Exposures
# NOTE: the notebook string started with a stray "sql_6 = " inside the triple quotes,
# which made the template invalid SQL. The stray prefix has been removed here.
sql_6 = """SELECT group_name, COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL AND TRIM(group_name) <> ''
GROUP BY group_name
ORDER BY outstanding_millions DESC
LIMIT 5"""

# VQ7: All Overdue Loan Accounts
sql_7 = """SELECT loan_account_number, borrower_name, s.sector_name,
       ROUND(l.total_outstanding / 1000000.0, 2) AS outstanding_millions,
       days_past_due, asset_classification
FROM loan_master AS l
JOIN sector_master AS s ON l.sector_code = s.sector_code
WHERE days_past_due > 0
ORDER BY days_past_due DESC, total_outstanding DESC"""

# VQ8: DPD Bucket Distribution
sql_8 = """SELECT CASE
           WHEN days_past_due = 0 THEN '0 (Current)'
           WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
           WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
           WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
           ELSE '90+'
       END AS dpd_bucket,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY CASE dpd_bucket WHEN '0 (Current)' THEN 1 WHEN '1-30' THEN 2
                         WHEN '31-60' THEN 3 WHEN '61-90' THEN 4 ELSE 5 END"""

# VQ9: Latest Rating Downgrades
sql_9 = """SELECT borrower_id, previous_rating, internal_rating,
       rating_direction, pd_estimate
FROM borrower_rating
WHERE rating_date = (SELECT MAX(rating_date) FROM borrower_rating)
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """SELECT reporting_date,
       ROUND(SUM(ecl_amount) / 1000000.0, 2) AS ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}

# =============================================================================
# TOOL DEFINITIONS
# `llm` and `evaluator_llm` are module-level globals assigned in main() before
# any tool is called, exactly as in the notebook.
# =============================================================================

llm = None
evaluator_llm = None


def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
1. Read the user's question carefully and identify its main business intent.
2. Compare the question with the descriptions of all available verified query templates.
3. If the question clearly matches the purpose of one verified template, select:
   - "route": "verified"
   - "query_id": the exact matching template ID.
4. A match does not require identical wording. Consider synonyms and equivalent business terms, such as:
   - Outstanding balance and loan exposure
   - NPA and non-performing assets
   - ECL and expected credit loss
   - DPD and days past due
   - Provision coverage and provision coverage ratio
5. Select a verified template only when its purpose answers the user's main question. Do not select a template based only on a shared keyword.
6. If the question does not clearly match any verified template, requires a different calculation, or asks for analysis not covered by the templates, select:
   - "route": "generated"
   - "query_id": null
7. Do not invent query IDs or select multiple templates.
8. Do not generate SQL or answer the user's business question. Your only task is classification.

{user_question}

{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

Write exactly one SQLite SELECT query and nothing else. Use only the supplied schema.
Never write or modify data. Do not use PRAGMA, ATTACH, DDL, DML, or multiple statements.
Return complete rows needed to answer the question. Monetary outputs should be in millions when practical.

{user_question}

{schema_context}

"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."

    )
    relevance_prompt = f"""
You are an expert SQL Relevance Validator for the Northbridge Bank Credit Risk Query Engine.

Your task is to evaluate whether the generated SQL query correctly answers the user's original natural language question.

### VALIDATION RULES

1. Compare the user's original question with the generated SQL query.
2. Check whether the SQL query addresses the correct business requirement.
3. Verify that the query uses appropriate tables and columns for the requested information.
4. Check whether the selected metrics, calculations, aggregations, and filters match the user's question.
5. Verify that date filters and reporting periods are appropriate when requested.
6. Ensure that GROUP BY, JOIN, WHERE, and ORDER BY clauses support the requested result.
7. Check that the query does not omit important conditions or introduce unrelated calculations.
8. Do not consider a query relevant simply because it contains similar keywords.
9. If the query answers the user's question correctly, mark it as relevant.
10. If the query answers a different question, misses important requirements, or uses incorrect business logic, mark it as not relevant.



{track_context}

{user_question}

{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            # The queries are wrapped in a subquery so that templates which already end in
            # LIMIT (VQ5, VQ6) do not produce an invalid "... LIMIT 10 LIMIT 0" statement.
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({candidate_sql}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
You are an expert SQL Developer for the Northbridge Bank Credit Risk Query Engine.

Your task is to fix the previously generated SQL query based on the validation error and produce a corrected SQL query that answers the user's original question.

### SQL REGENERATION RULES

1. Carefully analyze the original user question.
2. Review the failed SQL query and identify the reason it failed validation.
3. Read the validation error message and use it to correct the SQL.
4. Use only the tables and columns available in the provided database schema.
5. Ensure that table joins use the correct relationships and join conditions.
6. Correct incorrect column names, table names, filters, aggregations, and GROUP BY clauses.
7. Use appropriate SQL functions and calculations for the requested business metrics.
8. Ensure the query answers the original question, not a different business question.
9. Generate a single, complete, executable SQLite SELECT query.
10. Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or other data-modification statements.
11. Do not use multiple SQL statements.
12. Do not invent database tables, columns, or values.
13. If the validation error indicates a missing requirement, incorporate that requirement into the corrected query.
14. Return only the corrected SQL query without explanations or Markdown code fences.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

You are a Senior Credit Risk Analyst at Northbridge Bank.

Your task is to interpret SQL query results and provide a clear, accurate, and professional business response to the user's original question.

### RESPONSE GENERATION RULES

1. Understand the user's original business question before interpreting the results.
2. Use only the information available in the SQL query result.
3. Directly answer the user's question using the returned data.
4. Highlight important numbers, totals, percentages, trends, and risk indicators when available.
5. Explain the business meaning of the results in simple, professional language.
6. Identify significant credit-risk observations, such as high exposures, overdue loans, NPA accounts, rating downgrades, or increased ECL, only when supported by the data.
7. Use bullet points or a short table when they make the results easier to understand.
8. Include the number of records returned when useful.
9. Do not invent figures, assumptions, explanations, or recommendations that are not supported by the query results.
10. If the result is empty, clearly explain that no matching records were found.
11. If the result contains NULL or missing values, do not interpret them as zero unless the data supports that conclusion.
12. Keep the response concise, focused, and suitable for business stakeholders.
13. Do not include SQL code or technical implementation details.
14. Provide actionable business insights only when they can be supported by the returned data.

{user_question}

{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================


def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages in log['trace'].

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'trace': []          # Streamlit addition: replaces the notebook's print() output
    }

    def emit(message):
        """Stand-in for print(): keeps the pipeline stage messages for display/audit."""
        if verbose:
            log['trace'].append(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    emit(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    emit(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    emit(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    emit(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        emit(f"    Failed check: {gate.get('failed_check')}")
        emit(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        emit(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        emit(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            emit(f"    Retry failed check: {gate.get('failed_check')}")
            emit(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        emit(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    emit(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        emit(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    emit(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# =============================================================================
# STREAMLIT UI HELPERS
# =============================================================================

EXAMPLE_QUESTIONS = [
    "How much of our book is in real estate, and how much of that is non-performing?",
    "Show me the DPD aging profile of the portfolio.",
    "What are the top 10 largest loan exposures?",
    "What is the ECL trend across the reporting quarters?",
]


def safe_markdown(text: str) -> str:
    """Escape '$' so Streamlit does not render dollar amounts as LaTeX."""
    return str(text).replace("$", "\\$")


def format_confidence(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value) if value is not None else "n/a"


def build_audit_record(result: Dict[str, Any]) -> Dict[str, Any]:
    """Audit entry for one analytical output (question, route, SQL, gate result, confidence...)."""
    record = dict(result["log"])
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return record


def audit_to_jsonl(records: List[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, default=str) for r in records)


def render_result(result: Dict[str, Any]):
    """Display answer, confidence, SQL used, raw data, and pipeline trace for one run."""
    log = result["log"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", str(log["route"]).title())
    c2.metric("Query ID", log["query_id"] or "—")
    c3.metric("Confidence", format_confidence(log["confidence"]))
    c4.metric("Rows returned", "—" if log["row_count"] is None else log["row_count"])
    c5.metric("Retry used", "Yes" if log["retry_used"] else "No")

    if log["escalated"]:
        st.warning("🚨 **Escalated to human analyst.**  \n" + safe_markdown(log["narrative"]))
    else:
        st.subheader("Answer")
        st.markdown(safe_markdown(log["narrative"]))

    if log.get("match_reason"):
        st.caption(f"Routing reason: {log['match_reason']}")

    sql_tab, data_tab, trace_tab, audit_tab = st.tabs(
        ["SQL used", "Raw data", "Pipeline trace", "Audit record"]
    )

    with sql_tab:
        sql_to_show = log["executed_sql"] or log["candidate_sql"]
        if log["escalated"]:
            st.caption("This SQL did not pass validation and was NOT executed.")
        st.code(sqlparse.format(sql_to_show, reindent=True, keyword_case="upper"), language="sql")

    with data_tab:
        df = result["dataframe"]
        if df is None:
            st.info("No data returned (query was escalated before execution).")
        else:
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download result as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
            )

    with trace_tab:
        gate = log["gate_result"] or {}
        st.write(
            f"**Validation gate:** passed = `{gate.get('passed')}`, "
            f"failed check = `{gate.get('failed_check')}`, "
            f"relevance confidence = `{gate.get('relevance_confidence')}`"
        )
        st.text("\n".join(log.get("trace", [])) or "No trace recorded.")

    with audit_tab:
        st.json(json.loads(json.dumps({k: v for k, v in log.items()}, default=str)))


def run_and_record(question: str, conn):
    """Run the pipeline for one question and append it to the session audit trail."""
    result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
    st.session_state.audit_trail.append(build_audit_record(result))
    return result


def set_example(question: str):
    st.session_state.question_input = question


# =============================================================================
# MAIN APP
# =============================================================================


def main():
    global llm, evaluator_llm

    st.title("🏦 Northbridge Bank — Credit Risk Query Engine")
    st.caption(
        "Ask routine commercial lending portfolio questions in plain English. "
        "Recurring questions use pre-approved SQL templates; novel questions get validated, "
        "read-only generated SQL. Every answer shows its SQL, raw data and confidence, "
        "and failures escalate to a human analyst."
    )

    # ---- Session state ----------------------------------------------------
    st.session_state.setdefault("audit_trail", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("eval_results", None)

    # ---- Credentials and LLMs --------------------------------------------
    api_key, api_base = configure_credentials()
    if not api_key:
        st.error(
            "OPENAI_API_KEY not found. Add it to `.streamlit/secrets.toml`, set it as an "
            "environment variable, or provide it in a `config.json` file "
            "(keys: `OPENAI_API_KEY`, `OPENAI_API_BASE`)."
        )
        st.stop()
    llm, evaluator_llm = get_llms(api_key, api_base)

    # ---- Database ---------------------------------------------------------
    db_path = resolve_db_path()
    if db_path is None:
        st.error(
            "SQLite database not found. Place `credit_risk_portfolio.db` next to `app.py` "
            "(or set `DB_PATH` to its location)."
        )
        st.stop()
    conn = get_connection(str(db_path))

    ground_truth = load_ground_truth()

    # ---- Sidebar ----------------------------------------------------------
    with st.sidebar:
        st.header("System status")
        st.success("Database connected (read-only)")
        st.caption(f"`{db_path.name}`")
        st.caption(f"Query model: `{LLM_MODEL}` · Validator model: `{EVALUATOR_MODEL}`")

        with st.expander("Database tables"):
            for table_name in ["loan_master", "borrower_rating", "provisioning", "sector_master"]:
                try:
                    n = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table_name}", conn).iloc[0, 0]
                    st.write(f"**{table_name}**: {n:,} rows")
                except Exception as e:  # noqa: BLE001
                    st.write(f"**{table_name}**: unavailable ({e})")

        with st.expander("Verified query library"):
            for qid, entry in verified_query_library.items():
                st.markdown(f"**{qid}** — {entry['description']}")

        st.header("Example questions")
        examples = list(EXAMPLE_QUESTIONS)
        if ground_truth is not None and "User Query" in ground_truth.columns:
            for q in ground_truth["User Query"].dropna().tolist():
                if q not in examples:
                    examples.append(q)
        for i, q in enumerate(examples):
            st.button(q, key=f"example_{i}", on_click=set_example, args=(q,), use_container_width=True)

        st.header("Audit trail")
        st.caption(f"{len(st.session_state.audit_trail)} analytical output(s) this session")
        if st.session_state.audit_trail:
            st.download_button(
                "Download audit trail (JSONL)",
                data=audit_to_jsonl(st.session_state.audit_trail),
                file_name="audit_trail.jsonl",
                mime="application/json",
            )
            if st.button("Clear session history"):
                st.session_state.audit_trail = []
                st.session_state.last_result = None
                st.rerun()

    # ---- Main tabs ----------------------------------------------------------
    ask_tab, audit_tab, eval_tab = st.tabs(["Ask a question", "Audit trail", "Evaluation"])

    # ===== Ask tab =====
    with ask_tab:
        question = st.text_area(
            "Your question",
            key="question_input",
            height=100,
            placeholder="e.g. How much of our book is in real estate, and how much of that is non-performing?",
        )
        if st.button("Run query", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Running pipeline: classify → build SQL → validate → execute → respond..."):
                    try:
                        st.session_state.last_result = run_and_record(question.strip(), conn)
                    except Exception as e:  # noqa: BLE001
                        st.session_state.last_result = None
                        st.error(f"Pipeline error: {e}")

        if st.session_state.last_result is not None:
            render_result(st.session_state.last_result)

    # ===== Audit tab =====
    with audit_tab:
        trail = st.session_state.audit_trail
        if not trail:
            st.info("No queries have been run in this session yet.")
        else:
            summary = pd.DataFrame(
                [
                    {
                        "Timestamp": r["timestamp"],
                        "Question": r["user_question"],
                        "Route": r["route"],
                        "Query ID": r["query_id"],
                        "Confidence": r["confidence"],
                        "Retry used": r["retry_used"],
                        "Escalated": r["escalated"],
                        "Rows": r["row_count"],
                    }
                    for r in trail
                ]
            )
            st.dataframe(summary, use_container_width=True)
            for i, r in enumerate(reversed(trail), start=1):
                with st.expander(f"{r['timestamp']} — {r['user_question'][:90]}"):
                    st.json(json.loads(json.dumps(r, default=str)))

    # ===== Evaluation tab =====
    with eval_tab:
        st.subheader("Evaluation against ground truth")
        if ground_truth is None:
            st.info("`test_queries.csv` was not found next to `app.py`, so evaluation is unavailable.")
        else:
            st.dataframe(ground_truth, use_container_width=True)
            n_cases = st.number_input(
                "Number of test cases to run (from the top of the file)",
                min_value=1,
                max_value=len(ground_truth),
                value=min(5, len(ground_truth)),
                step=1,
            )
            if st.button("Run evaluation"):
                evaluation_rows = []
                progress = st.progress(0.0)
                for i, (_, gt) in enumerate(ground_truth.head(int(n_cases)).iterrows()):
                    try:
                        tr = run_and_record(gt["User Query"], conn)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Test case {gt['Test Case']} failed: {e}")
                        tr = {"route": None, "query_id": None, "confidence": None, "row_count": None}

                    evaluation_rows.append({
                        'Test Case': gt['Test Case'],
                        'Expected Route': gt['Expected Route'],
                        'Actual Route': tr['route'],
                        'Route Match': tr['route'] == gt['Expected Route'],
                        'Expected Query ID': gt['Expected Query ID'],
                        'Actual Query ID': tr['query_id'],
                        'Query ID Match': (
                            pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                        ) or tr['query_id'] == gt['Expected Query ID'],
                        'Confidence': tr['confidence'],
                        'Rows Returned': tr['row_count']
                    })
                    progress.progress((i + 1) / int(n_cases))
                progress.empty()
                st.session_state.eval_results = pd.DataFrame(evaluation_rows)

            evaluation_df = st.session_state.eval_results
            if evaluation_df is not None:
                path_accuracy = evaluation_df['Route Match'].mean() * 100

                verified = evaluation_df['Expected Route'].astype(str).str.strip().str.lower() == 'verified'
                query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100  # NaN if no verified cases

                average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

                m1, m2, m3 = st.columns(3)
                m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
                m2.metric("Selected Query Accuracy", "n/a" if pd.isna(query_accuracy) else f"{query_accuracy:.1f}%")
                m3.metric("Average Confidence Score", "n/a" if pd.isna(average_confidence) else f"{average_confidence:.2f}")
                st.dataframe(evaluation_df, use_container_width=True)
                st.caption(
                    "Confidence is the validation gate's LLM relevance score (0–1); it is not a "
                    "statistically calibrated probability of correctness."
                )


if __name__ == "__main__":
    main()
