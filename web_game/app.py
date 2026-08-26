"""FastAPI entrypoint for the local/browser Tichu prototype."""

from __future__ import annotations

import os
from pathlib import Path
import secrets
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gym_agents.model_c import ModelCAgent
from .controller import WebGame, WebGameError


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name("static")
CHECKPOINT = Path(os.environ.get("TICHU_MODEL_CHECKPOINT", ROOT / "models/model-c-v2/model-c.pt"))
DEVICE = os.environ.get("TICHU_MODEL_DEVICE", "auto")

app = FastAPI(title="Tichu: 사람 2명 vs Model C")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@dataclass
class Room:
    game: WebGame
    seat_tokens: dict[int, str]


rooms: dict[str, Room] = {}


class JoinRequest(BaseModel):
    seat: int


class ActionRequest(BaseModel):
    action_id: str


class PromptRequest(BaseModel):
    choice: str | int


class CreateRoomRequest(BaseModel):
    mode: str = "single"


def get_room(room_id: str) -> Room:
    try:
        return rooms[room_id]
    except KeyError as error:
        raise HTTPException(404, "방을 찾을 수 없습니다.") from error


def authorize(room_id: str, seat: int, token: str) -> WebGame:
    room = get_room(room_id)
    if seat not in room.game.human_seats or not secrets.compare_digest(
        room.seat_tokens.get(seat, ""), token
    ):
        raise HTTPException(403, "이 좌석의 초대 링크가 아닙니다.")
    return room.game


def as_http_error(error: WebGameError):
    raise HTTPException(400, str(error)) from error


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/rooms")
def create_room(request: CreateRoomRequest | None = None):
    if not CHECKPOINT.exists():
        raise HTTPException(500, "Model C 체크포인트를 찾지 못했습니다: {}".format(CHECKPOINT))
    mode = "single" if request is None else request.mode
    if mode not in {"single", "friends"}:
        raise HTTPException(400, "mode는 single 또는 friends여야 합니다.")
    human_seats = frozenset({0} if mode == "single" else {0, 2})
    room_id = secrets.token_urlsafe(6)
    tokens = {seat: secrets.token_urlsafe(24) for seat in human_seats}
    rooms[room_id] = Room(
        game=WebGame(
            lambda: ModelCAgent(CHECKPOINT, device=DEVICE),
            human_seats=human_seats,
        ),
        seat_tokens=tokens,
    )
    return {
        "room_id": room_id,
        "mode": mode,
        "host_token": tokens[0],
        "friend_token": tokens.get(2),
        "seats": sorted(human_seats),
    }


@app.post("/api/rooms/{room_id}/join")
def join_room(room_id: str, request: JoinRequest, token: str):
    authorize(room_id, request.seat, token)
    return {"room_id": room_id, "seat": request.seat}


@app.get("/api/rooms/{room_id}/state/{seat}")
def state(room_id: str, seat: int, token: str):
    try:
        return authorize(room_id, seat, token).view_for(seat)
    except WebGameError as error:
        as_http_error(error)


@app.post("/api/rooms/{room_id}/state/{seat}/action")
def action(room_id: str, seat: int, request: ActionRequest, token: str):
    game = authorize(room_id, seat, token)
    try:
        game.play(seat, request.action_id)
        return game.view_for(seat)
    except WebGameError as error:
        as_http_error(error)


@app.post("/api/rooms/{room_id}/state/{seat}/prompt")
def prompt(room_id: str, seat: int, request: PromptRequest, token: str):
    game = authorize(room_id, seat, token)
    try:
        game.choose_prompt(seat, request.choice)
        return game.view_for(seat)
    except WebGameError as error:
        as_http_error(error)
