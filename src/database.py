"""SQLite database utilities for census income analysis.

Demonstrates SQL proficiency: CTEs, window functions, CASE expressions,
aggregations with HAVING, and subqueries.
"""

import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "census.db")


def create_database(data: pd.DataFrame, db_path: str = None) -> str:
    """Load census data into a SQLite database."""
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    data.to_sql("census", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_income ON census(income)")
    conn.commit()
    conn.close()
    return db_path


def run_query(query: str, db_path: str = None) -> pd.DataFrame:
    """Execute a SQL query and return results."""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


QUERIES = {
    "income_by_education_occupation": """
        SELECT
            education_level,
            occupation,
            COUNT(*) AS total,
            SUM(CASE WHEN income = '>50K' THEN 1 ELSE 0 END) AS high_income,
            ROUND(100.0 * SUM(CASE WHEN income = '>50K' THEN 1 ELSE 0 END) / COUNT(*), 2)
                AS pct_high_income
        FROM census
        GROUP BY education_level, occupation
        HAVING total > 100
        ORDER BY pct_high_income DESC
        LIMIT 20
    """,

    "demographic_income_profile": """
        WITH income_stats AS (
            SELECT
                sex,
                race,
                income,
                COUNT(*) AS cnt,
                ROUND(AVG(age), 1) AS avg_age,
                ROUND(AVG("education-num"), 1) AS avg_education,
                ROUND(AVG("hours-per-week"), 1) AS avg_hours
            FROM census
            GROUP BY sex, race, income
        )
        SELECT
            sex,
            race,
            income,
            cnt,
            avg_age,
            avg_education,
            avg_hours
        FROM income_stats
        ORDER BY sex, race, income
    """,

    "age_income_distribution": """
        SELECT
            CASE
                WHEN age < 25 THEN 'Under 25'
                WHEN age BETWEEN 25 AND 34 THEN '25-34'
                WHEN age BETWEEN 35 AND 44 THEN '35-44'
                WHEN age BETWEEN 45 AND 54 THEN '45-54'
                WHEN age BETWEEN 55 AND 64 THEN '55-64'
                ELSE '65+'
            END AS age_group,
            COUNT(*) AS total,
            SUM(CASE WHEN income = '>50K' THEN 1 ELSE 0 END) AS high_income,
            ROUND(100.0 * SUM(CASE WHEN income = '>50K' THEN 1 ELSE 0 END) / COUNT(*), 2)
                AS pct_high_income
        FROM census
        GROUP BY age_group
        ORDER BY MIN(age)
    """,

    "capital_gains_percentiles": """
        WITH ranked AS (
            SELECT
                "capital-gain",
                income,
                NTILE(10) OVER (ORDER BY "capital-gain") AS decile
            FROM census
            WHERE "capital-gain" > 0
        )
        SELECT
            decile,
            MIN("capital-gain") AS min_gain,
            MAX("capital-gain") AS max_gain,
            COUNT(*) AS count,
            ROUND(100.0 * SUM(CASE WHEN income = '>50K' THEN 1 ELSE 0 END) / COUNT(*), 2)
                AS pct_high_income
        FROM ranked
        GROUP BY decile
        ORDER BY decile
    """,

    "marital_education_cross": """
        SELECT
            "marital-status",
            education_level,
            COUNT(*) AS total,
            ROUND(100.0 * SUM(CASE WHEN income = '>50K' THEN 1 ELSE 0 END) / COUNT(*), 2)
                AS pct_high_income,
            ROUND(AVG("hours-per-week"), 1) AS avg_hours,
            ROUND(AVG("capital-gain"), 0) AS avg_capital_gain
        FROM census
        GROUP BY "marital-status", education_level
        HAVING total >= 50
        ORDER BY pct_high_income DESC
        LIMIT 20
    """,
}
