"""Bot turn markers for replies the transport never speaks.

The output transport marks the bot's turn from the audio it plays, so a
``skip_tts`` reply (a text-mode eval) would never end its turn: flow
post-actions such as end_conversation and the idle timer depend on it.
"""

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UnspokenReplyProcessor(FrameProcessor):
    """Broadcasts bot started/stopped speaking around an unspoken reply."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame) and frame.skip_tts:
            await self._set_speaking(True)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame) and frame.skip_tts:
            await self.push_frame(frame, direction)
            await self._set_speaking(False)
        elif isinstance(frame, InterruptionFrame):
            await self.push_frame(frame, direction)
            await self._set_speaking(False)
        else:
            await self.push_frame(frame, direction)

    async def _set_speaking(self, speaking: bool) -> None:
        if speaking == self._speaking:
            return
        self._speaking = speaking
        await self.broadcast_frame(
            BotStartedSpeakingFrame if speaking else BotStoppedSpeakingFrame
        )
