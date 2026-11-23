"""equip-voice：本项目精简语音交互入口。

用法示例：
    equip-voice                          # 常驻实时对话（默认启用本地唤醒）
    equip-voice --no-wake                # 调试：VAD 常开，不需要唤醒词
    equip-voice --list-devices           # 查看麦克风/扬声器序号
    equip-voice --input ask.wav --output reply.wav   # 离线 WAV 链路
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path

from cat_assistant.bootstrap import build_demo_app, load_env_file
from cat_assistant.voice.config import VoiceConfig, load_voice_config


def _apply_overrides(config: VoiceConfig, args: argparse.Namespace) -> VoiceConfig:
    audio = config.audio
    asr = config.asr
    tts = config.tts
    wake = config.wake
    session = config.session

    if args.no_wake:
        wake = replace(wake, provider="none")
    if args.wake_word:
        wake = replace(wake, word=args.wake_word)
    if args.no_ack:
        wake = replace(wake, ack="none")
    if args.input_device:
        audio = replace(audio, input_device=args.input_device)
    if args.output_device:
        audio = replace(audio, output_device=args.output_device)
    if args.asr_url:
        asr = replace(asr, base_url=args.asr_url)
    if args.tts_url:
        tts = replace(tts, base_url=args.tts_url)
    if args.ref_audio is not None:
        tts = replace(tts, ref_audio=args.ref_audio)
    if args.ref_text is not None:
        tts = replace(tts, ref_text=args.ref_text)
    if args.machine_id:
        session = replace(session, machine_id=args.machine_id)
    if args.operator_id:
        session = replace(session, operator_id=args.operator_id)
    if args.session_id:
        session = replace(session, session_id=args.session_id)
    return replace(config, audio=audio, asr=asr, tts=tts, wake=wake, session=session)


def _device_label(value: object) -> str:
    return "默认" if value in (None, "", "default") else str(value)


def _print_devices() -> int:
    try:
        from cat_assistant.voice.audio import list_devices
    except ImportError as exc:
        print(f"未安装语音依赖：{exc}", file=sys.stderr)
        return 2
    try:
        devices = list_devices()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not devices:
        print("未发现音频设备。")
        return 0
    print("音频设备：")
    for item in devices:
        caps = []
        if item["inputs"]:
            caps.append(f"输入 {item['inputs']}")
        if item["outputs"]:
            caps.append(f"输出 {item['outputs']}")
        caps_text = " / ".join(caps) or "无"
        print(
            f"  [{item['index']}] {item['name']}（{caps_text}，"
            f"{item['sample_rate']} Hz）"
        )
    print("用 --input-device / --output-device 指定序号或名称。")
    return 0


async def _run(args: argparse.Namespace, config: VoiceConfig) -> int:
    try:
        from cat_assistant.voice.audio import WavFileSource, WavSink
        from cat_assistant.voice.loop import VoiceLoop, offline_config
    except ImportError as exc:
        print(
            "未安装语音依赖（numpy/sounddevice）："
            "pip install 'equip-ai-agent[voice]'，"
            f"原始错误：{exc}",
            file=sys.stderr,
        )
        return 2

    offline = bool(args.input)
    if offline:
        config = offline_config(config)
        output_path = Path(args.output) if args.output else Path(args.input).with_suffix(".reply.wav")
        source = WavFileSource(
            args.input,
            sample_rate=config.audio.sample_rate,
            frame_ms=config.audio.frame_ms,
        )
        sink = WavSink(output_path, sample_rate=config.audio.output_sample_rate)
    else:
        output_path = None
        source = None
        sink = None

    app = build_demo_app(config_path=args.model_config_path)
    loop = VoiceLoop(config, app, source=source, sink=sink)
    print(
        "语音交互配置：\n"
        f"  ASR   {config.asr.base_url}\n"
        f"  TTS   {config.tts.base_url}\n"
        f"  唤醒  {config.wake.provider if config.wake.enabled else '关闭（VAD 常开）'}"
        f"  关键词「{config.wake.word}」\n"
        f"  会话  {loop.session_id}\n"
        f"  设备  麦克风 {_device_label(config.audio.input_device)} / "
        f"扬声器 {_device_label(config.audio.output_device)}",
        flush=True,
    )
    if offline:
        print(f"离线模式：{args.input} → {output_path}", flush=True)
    else:
        print("开始对话，按 Ctrl+C 退出。", flush=True)

    await loop.run()
    if offline and output_path is not None:
        print(f"回复音频已写入：{output_path}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="equip-voice",
        description="MechDiagBot 精简语音交互（本地唤醒 + Qwen ASR + VoxCPM2 TTS）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="语音 JSON 配置（默认依次查找 $CAT_VOICE_CONFIG、"
        "runtime/voice-config.json、config/voice-config.json）",
    )
    parser.add_argument(
        "--model-config-path",
        type=Path,
        default=Path("runtime/runtime-config.json"),
        help="本项目模型/运行时配置（默认 runtime/runtime-config.json）",
    )
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    parser.add_argument("--input", help="离线模式：输入 WAV 文件")
    parser.add_argument("--output", help="离线模式：回复 WAV 输出路径")
    parser.add_argument("--no-wake", action="store_true", help="关闭唤醒词，VAD 常开")
    parser.add_argument("--wake-word", help="覆盖唤醒词，例如「你好，阿克西娅」")
    parser.add_argument("--no-ack", action="store_true", help="唤醒后不播放确认音")
    parser.add_argument("--input-device", help="麦克风设备序号或名称")
    parser.add_argument("--output-device", help="扬声器设备序号或名称")
    parser.add_argument("--asr-url", help="覆盖 Qwen ASR 服务地址")
    parser.add_argument("--tts-url", help="覆盖 VoxCPM2 TTS 服务地址")
    parser.add_argument("--ref-audio", help="VoxCPM2 参考音色 WAV 路径")
    parser.add_argument("--ref-text", help="参考音色对应文本")
    parser.add_argument("--machine-id", help="覆盖设备 ID")
    parser.add_argument("--operator-id", help="覆盖操作员 ID")
    parser.add_argument("--session-id", help="覆盖语音会话 ID")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser


def main() -> int:
    load_env_file()
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.list_devices:
        return _print_devices()
    try:
        config = _apply_overrides(load_voice_config(args.config), args)
    except ValueError as exc:
        print(f"语音配置错误：{exc}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(_run(args, config))
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"语音交互启动失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
