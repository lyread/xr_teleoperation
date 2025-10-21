import numpy as np
import matplotlib.pyplot as plt

class SmoothFilter:
    def __init__(self, data_size=2, window_size=5):
        self._data_size = data_size
        self._window_size = window_size
        self._data_queue = []
        self._filtered_data = np.zeros(data_size)

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        # 跳过重复数据，节省计算量
        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return
        
        # 如果超过窗口长度，移除最早一帧
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        # 添加新数据并更新滤波结果
        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    def _apply_filter(self):
        # 简单平均滤波（也可以换成加权平均）
        return np.mean(self._data_queue, axis=0)

    @property
    def filtered_data(self):
        return self._filtered_data


# -------------------------
# 模拟测试
# -------------------------
if __name__ == "__main__":
    # 创建滤波器，窗口大小5帧
    filter = SmoothFilter(data_size=2, window_size=5)

    # 模拟输入数据（含噪声）
    np.random.seed(42)
    time_steps = 50
    raw_actions = np.zeros((time_steps, 2))
    for i in range(time_steps):
        base = i / 50  # 递增趋势
        noise = np.random.uniform(-0.1, 0.1, 2)
        raw_actions[i] = [base + noise[0], base + noise[1]]

    # 应用滤波器
    smoothed_actions = []
    for a in raw_actions:
        filter.add_data(a)
        smoothed_actions.append(filter.filtered_data.copy())
    smoothed_actions = np.array(smoothed_actions)

    # 绘图对比
    plt.figure(figsize=(10, 5))
    plt.plot(raw_actions[:, 0], label="Raw Left", alpha=0.5, linestyle='--')
    plt.plot(smoothed_actions[:, 0], label="Smoothed Left", linewidth=2)
    plt.plot(raw_actions[:, 1], label="Raw Right", alpha=0.5, linestyle='--')
    plt.plot(smoothed_actions[:, 1], label="Smoothed Right", linewidth=2)
    plt.title("SmoothFilter.add_data() Filtering Effect")
    plt.xlabel("Time step")
    plt.ylabel("Action Value")
    plt.legend()
    plt.grid(True)
    plt.show()
