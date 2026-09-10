"""SQLAlchemy ORM models mirroring db/schema.sql.

schema.sql is authoritative (§6). These classes must follow it, not the reverse.
Hot paths use raw SQL with PostGIS functions rather than the ORM (§3).
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Scene(Base):
    __tablename__ = "scenes"
    __abstract__ = True


class Detection(Base):
    __tablename__ = "detections"
    __abstract__ = True


class DetectionFactor(Base):
    __tablename__ = "detection_factors"
    __abstract__ = True


class DriftRun(Base):
    __tablename__ = "drift_runs"
    __abstract__ = True


class DriftState(Base):
    __tablename__ = "drift_states"
    __abstract__ = True


class Vessel(Base):
    __tablename__ = "vessels"
    __abstract__ = True


class AisPosition(Base):
    __tablename__ = "ais_positions"
    __abstract__ = True


class AisTrack(Base):
    __tablename__ = "ais_tracks"
    __abstract__ = True


class Attribution(Base):
    __tablename__ = "attributions"
    __abstract__ = True


class Dossier(Base):
    __tablename__ = "dossiers"
    __abstract__ = True


class DemoScenario(Base):
    __tablename__ = "demo_scenarios"
    __abstract__ = True
