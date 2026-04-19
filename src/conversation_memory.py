from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Message:
    role: str    # "user" | "assistant"
    content: str


class ConversationMemory:
    """基于滑动窗口的多轮对话记忆，按 conversation_id 隔离"""

    def __init__(self, window_size: int = 5) -> None:
        # 保留最近 window_size 轮（每轮 = 1问 + 1答）
        self.window_size = window_size
        self._store: Dict[str, List[Message]] = defaultdict(list)

    def add(self, conversation_id: str, role: str, content: str) -> None:
        self._store[conversation_id].append(Message(role=role, content=content))

    def get_formatted(self, conversation_id: str) -> str:
        """返回格式化的历史文本，供 prompt 拼接"""
        messages = self._store[conversation_id]
        recent = messages[-(self.window_size * 2):]
        if not recent:
            return ""
        lines = [
            f"{'用户' if m.role == 'user' else '助手'}：{m.content}"
            for m in recent
        ]
        return "\n".join(lines)

    def clear(self, conversation_id: str) -> None:
        self._store[conversation_id] = []