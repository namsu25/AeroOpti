"""Pydantic request/response models for the AeroOpti API."""

from pydantic import BaseModel


class Weights(BaseModel):
    fuel: float = 1.0
    time: float = 0.5
    risk: float = 2.0
    airspace_fee: float = 0.3
    traffic: float = 0.5


class OptimizeRequest(BaseModel):
    origin: str
    destination: str
    aircraft: str
    payload_pct: float = 75.0
    weights: Weights = Weights()


class Route(BaseModel):
    name: str
    points: list[list[float]]
    fuel_kg: float
    time_h: float
    risk: float
    congestion: float
    cost_usd: float
    co2_kg: float
    color: str


class OptimizeResponse(BaseModel):
    routes: list[Route]
    best_index: int
