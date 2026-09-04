"""Celery worker and beat entry point for Rubituci background learning."""

import asyncio

from celery import Celery
from celery.schedules import crontab

from api.config import settings


celery_app = Celery("rubituci", broker=settings.CELERY_BROKER_URL, backend=settings.CELERY_RESULT_BACKEND)
celery_app.conf.timezone = "America/Sao_Paulo"
celery_app.conf.beat_schedule = {
    "rubituci-computational-sleep": {
        "task": "worker.computational_sleep",
        "schedule": crontab(hour=3, minute=0),
    }
}


@celery_app.task(name="worker.computational_sleep", bind=True, max_retries=2)
def computational_sleep(self):
    async def run():
        from api.database import db_manager
        from api.models import ModelGeneration
        from learning.sleep import SleepCycle
        from sqlalchemy import select

        async with db_manager.session() as db:
            current = (await db.execute(
                select(ModelGeneration).where(ModelGeneration.status == "promoted").order_by(ModelGeneration.generation_number.desc())
            )).scalar_one_or_none()
            report = await SleepCycle(db, current.generation_number if current else 1).run(allow_web_research=True)
            return report.__dict__
    try:
        return asyncio.run(run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=300)
