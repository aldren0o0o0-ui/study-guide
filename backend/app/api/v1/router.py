from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.modules.analysis.router import analyses_router
from app.modules.auth.router import router as auth_router
from app.modules.chat.router import chat_router
from app.modules.documents.router import documents_router
from app.modules.exams.router import exams_router
from app.modules.planner.router import planner_router
from app.modules.voice_practice.ws_router import voice_practice_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(auth_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(analyses_router)
api_v1_router.include_router(exams_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(planner_router)
api_v1_router.include_router(voice_practice_router)
