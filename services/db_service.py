from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, func, text
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "scans.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    stress_score = Column(Float, nullable=False)
    fatigue_score = Column(Float, nullable=False)
    eye_strain = Column(Float, nullable=False)
    recovery_score = Column(Float, nullable=False, default=0)
    wellness_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    recovery_need = Column(String(24), nullable=False)
    summary = Column(Text, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(scans)").fetchall()}
        if "recovery_score" not in columns:
            connection.execute(text("ALTER TABLE scans ADD COLUMN recovery_score FLOAT NOT NULL DEFAULT 0"))


def save_scan(report: dict) -> None:
    init_db()
    scan = Scan(
        stress_score=float(report.get("stress_score", 0)),
        fatigue_score=float(report.get("fatigue_score", 0)),
        eye_strain=float(report.get("eye_strain", 0)),
        recovery_score=float(report.get("recovery_score", 0)),
        wellness_score=float(report.get("wellness_score", 0)),
        confidence=float(report.get("confidence", 0)),
        recovery_need=str(report.get("recovery_need", "Medium")),
        summary=str(report.get("wellness_summary", ""))[:1200],
    )
    with SessionLocal() as session:
        session.add(scan)
        session.commit()


def recent_scans(limit: int = 30) -> list[dict]:
    init_db()
    with SessionLocal() as session:
        rows = session.query(Scan).order_by(Scan.created_at.desc()).limit(limit).all()
    return [
        {
            "id": row.id,
            "created_at": row.created_at,
            "stress_score": row.stress_score,
            "fatigue_score": row.fatigue_score,
            "eye_strain": row.eye_strain,
            "recovery_score": row.recovery_score,
            "wellness_score": row.wellness_score,
            "confidence": row.confidence,
            "recovery_need": row.recovery_need,
            "summary": row.summary,
        }
        for row in reversed(rows)
    ]
