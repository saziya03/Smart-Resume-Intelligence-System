import fitz  # PyMuPDF

resume_text = """SAZIYA ALVI
+91-6306632089 | saziyaalvi7@gmail.com | linkedin.com/in/saziya03 | github.com/saziya03

SUMMARY
Detail-oriented MCA student with hands-on project experience in Python, SQL, Excel, and Power BI. Skilled in data cleaning, EDA, SQL querying, and interactive dashboard development. Built 4 analytics projects to uncover trends and translate data into actionable business insights.

EDUCATION
Master of Computer Applications (MCA), JSS Academy of Technical Education, Noida - 2025-Present
Bachelor of Science (B.Sc.), CSJMU, Kanpur - 2021-2024
Senior Secondary (Class XII), Unnao Public Inter College, UP Board - 2021 - 81%

TECHNICAL SKILLS
Programming: Python, Pandas, NumPy
SQL & Database: SQL, MySQL, Joins, CTEs, Subqueries, Window Functions
Data Analysis: Data Cleaning, Data Wrangling, EDA, Statistics
Excel: Pivot Tables, XLOOKUP, VLOOKUP, SUMIF, COUNTIF
Visualization: Power BI, Power Query, DAX
Tools: Git, GitHub, Jupyter, VS Code

PROJECTS
1. Customer Churn Analysis [GitHub] - Python, SQL, Pandas
- Analyzed 10K+ customer records using Python and Pandas to identify churn patterns and perform data cleaning and EDA.
- Visualized customer churn trends to identify key factors associated with customer attrition.

2. Blinkit Grocery Sales Dashboard [GitHub] - Power BI, DAX
- Developed an interactive Power BI dashboard to analyze sales, revenue, and outlet performance across 8,523+ records.
- Created DAX measures and Power Query transformations to track KPIs and business trends.

3. Financial Loan Analysis Dashboard [GitHub] - Advanced Excel
- Analyzed 38K+ loan applications using Pivot Tables, Slicers, and KPIs to examine loan performance and default patterns.
- Developed an interactive dashboard to track loan applications, funded amounts, and Good vs. Bad Loans.

4. PhonePe Transaction Insights Dashboard [GitHub] - Power BI, DAX
- Developed an interactive Power BI dashboard to analyze 50K+ UPI transactions by volume, value, and state.
- Used DAX measures and visualizations to identify transaction trends and compare state-wise performance.

CERTIFICATIONS & JOB SIMULATIONS
- Deloitte Australia - Data Analytics Job Simulation, Forage
"""

doc = fitz.open()
page = doc.new_page(width=595, height=842)  # A4 size
# Text ko page me daalo, auto-wrap ke saath
page.insert_textbox(fitz.Rect(50, 50, 545, 792), resume_text, fontsize=10, fontname="helv", align=0)
doc.save("data/resumes/sample_resume_fixed.pdf")
doc.close()