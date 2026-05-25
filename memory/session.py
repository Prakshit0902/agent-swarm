from __future__ import annotations
import sqlite3, json, time
from config.settings import settings

class SessionMemory:
    def __init__(self, session_id: str):
        self.sid = session_id
        self.db = sqlite3.connect(settings.sessions_db)
        self.db.execute("""CREATE TABLE IF NOT EXISTS turns(
          sid TEXT, ts REAL, role TEXT, agent TEXT, content TEXT)""")
        self.db.commit()
    def log(self, role, agent, content):
        self.db.execute("INSERT INTO turns VALUES(?,?,?,?,?)",
                        (self.sid, time.time(), role, agent,
                         content if isinstance(content,str) else json.dumps(content)))
        self.db.commit()
    def history(self, limit=40):
        rows = self.db.execute("SELECT role, content FROM turns WHERE sid=? ORDER BY ts DESC LIMIT ?",
                               (self.sid, limit)).fetchall()
        return [{"role":r, "content":c} for r,c in reversed(rows)]
