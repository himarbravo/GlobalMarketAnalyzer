import sqlite3
import datetime
import os

DB_NAME = "brain.db"

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.cursor = self.conn.cursor()
        self.init_db()

    def init_db(self):
        # Tabla de Señales (Memoria del Analista) - Mantenemos esto para historial de análisis
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                ticker TEXT,
                signal TEXT, -- BUY / SELL / HOLD
                confidence REAL,
                price REAL,
                rationale TEXT
            )
        ''')
        self.conn.commit()

    def log_signal(self, ticker, signal, confidence, price, rationale):
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute('''
            INSERT INTO signals (date, ticker, signal, confidence, price, rationale)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date, ticker, signal, confidence, price, rationale))
        self.conn.commit()

    def get_signals_history(self):
        self.cursor.execute('SELECT date, ticker, signal, confidence, price, rationale FROM signals ORDER BY date DESC')
        return self.cursor.fetchall()
