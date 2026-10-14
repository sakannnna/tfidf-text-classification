"""训练日志的双路输出。"""

class Tee:
    """将 sklearn 每轮训练输出同时显示并保存。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()
