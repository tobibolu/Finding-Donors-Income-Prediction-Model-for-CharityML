-- =============================================================================
-- Analytical SQL Queries for Census Income Analysis
-- Demonstrates: CTEs, window functions, CASE, HAVING, subqueries
-- =============================================================================

-- 1. Income rates by education and occupation (with GROUP BY + HAVING)
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
LIMIT 20;


-- 2. Demographic profile with CTE
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
SELECT * FROM income_stats
ORDER BY sex, race, income;


-- 3. Capital gains decile analysis with window functions (NTILE)
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
ORDER BY decile;


-- 4. Age group income distribution with CASE
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
ORDER BY MIN(age);


-- 5. Marital-education cross-tabulation with aggregates
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
LIMIT 20;
