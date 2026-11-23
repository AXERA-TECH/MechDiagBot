"""``cat_assistant.voice``：精简语音交互（本地唤醒 + Qwen ASR + VoxCPM2 TTS）。"""

from cat_assistant.voice.config import VoiceConfig, load_voice_config

__all__ = ["VoiceConfig", "load_voice_config"]
