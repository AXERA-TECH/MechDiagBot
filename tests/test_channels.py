import asyncio

from cat_assistant.channels import ChannelBridge, ChannelMessage, OneBotV11Channel, WeChatWebhookChannel
from cat_assistant.domain.models import AssistantResponse, ResponseCategory


def test_onebot_group_message_normalizes_and_sends(monkeypatch):
    channel = OneBotV11Channel("http://127.0.0.1:5700")
    message = asyncio.run(channel.handle_update({
        "post_type": "message", "message_type": "group", "group_id": 42,
        "user_id": 7, "message_id": 9, "message": "你好",
    }))
    assert message is not None
    assert message.chat_id == "group:42"
    assert message.sender_id == "7"


def test_wechat_common_payload():
    channel = WeChatWebhookChannel("https://example.invalid")
    message = asyncio.run(channel.handle_update({"conversation_id": "c1", "from_user": "u1", "content": "hi"}))
    assert message is not None
    assert (message.chat_id, message.sender_id, message.text) == ("c1", "u1", "hi")


class FakeApp:
    async def handle(self, utterance):
        self.utterance = utterance
        return AssistantResponse("收到", ResponseCategory.INFORMATION)


class FakeChannel:
    name = "fake"
    async def send_text(self, chat_id, text, *, reply_to=None):
        self.sent = (chat_id, text, reply_to)

    async def handle_update(self, payload):
        raise NotImplementedError


def test_bridge_scopes_session_and_operator():
    app, channel = FakeApp(), FakeChannel()
    bridge = ChannelBridge(app, machine_id="m1")
    asyncio.run(bridge.dispatch(channel, ChannelMessage("fake", "chat", "user", "hello", "msg")))
    assert app.utterance.session_id == "fake:chat"
    assert app.utterance.operator_id == "fake:user"
    assert channel.sent == ("chat", "收到", "msg")
