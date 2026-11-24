"""서빙 MAB는 영속 노출/클릭 집계를 읽는다. 평가·시뮬레이션은 별도 메모리 통계를 쓴다."""

from src.recommendation.reranking.mab import ArmStats


class JournalBanditStats:
    def __init__(self, log, model_version: str):
        self.log, self.model_version = log, model_version

    def get(self, arms):
        counts = self.log.feedback_counts(self.model_version)
        return {arm: ArmStats(counts.get(arm, {}).get("impressions", 0),
                              counts.get(arm, {}).get("clicks", 0)) for arm in arms}

    def record(self, arm, clicked):
        pass  # 실제 노출/클릭 API만 journal을 갱신한다.

    def record_shown(self, arm):
        pass

    def record_click(self, arm):
        pass
