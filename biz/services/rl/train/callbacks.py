import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class PlottingCallback(BaseCallback):
    """
    학습 중 Episode Reward를 기록하고, 종료 시 matplotlib을 통해 차트를 생성하는 콜백
    """
    def __init__(self, verbose=0, save_path="learning_curve.png"):
        super(PlottingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # 각 스텝마다 done이 발생한 환경에 대한 리워드를 수집 (Monitor Wrapper 필수)
        for info in self.locals.get("infos", []):
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        if len(self.episode_rewards) > 0:
            plt.figure(figsize=(10, 5))
            
            # 이동 평균(Moving Average) 계산
            window_size = min(20, len(self.episode_rewards))
            if window_size > 0:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            
            plt.plot(self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
            if window_size > 0:
                plt.plot(range(window_size - 1, len(self.episode_rewards)), moving_avg, color='red', label=f'{window_size}-Episode Moving Avg')
                
            plt.title('PPO Learning Curve')
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.save_path)
            plt.close()
            print(f"[차트] 학습 곡선 차트 저장 완료: {self.save_path}")
        else:
            print("에피소드 정보가 충분하지 않아 차트를 그릴 수 없습니다.")
