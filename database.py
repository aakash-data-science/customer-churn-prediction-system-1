import sqlite3


conn = sqlite3.connect(

    "database/churn_history.db",

    check_same_thread=False

)

cursor = conn.cursor()


cursor.execute(

'''

CREATE TABLE IF NOT EXISTS prediction_history(

id INTEGER PRIMARY KEY AUTOINCREMENT,

prediction TEXT,

churn_probability REAL,

retention_probability REAL,

risk_level TEXT,

timestamp TEXT

)

'''

)

conn.commit()