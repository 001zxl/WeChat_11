from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .audit import AuditStore
from .brain.llm import ModelRouter
from .brain.memory import MemoryStore
from .brain.persona import Persona, build_system_prompt, load_persona
from .brain.ecommerce import EcommerceService
from .brain.rag import EmbeddingClient, KnowledgeBase
from .brain.tools import ToolBox
from .brain.vision import VisionClient
from .config import Config
from .executor.sender import SafeSender
from .perception.chat import lines_from_ocr, settle_capture
from .perception.layout import LayoutCalculator, Regions
from .perception.dual import PixelSentinel, downscale_for_ocr, lines_from_ocr_colored
from .perception.media import detect_chat_images, extract_items
from .perception.ocr import OcrEngine
from .perception.unread import click_point, find_red_badges
from .platform.base import MessageItem, Platform
from .policy import (
    Guardrails,
    RateLimiter,
    Whitelist,
    group_allows,
    in_quiet_hours,
    is_group_name,
    normalize_name,
)


_SYSTEM_PATTERNS = (
    "拍了拍", "撤回了一条消息", "以下是新消息", "该消息已过期",
    "对方正在输入", "查看更多消息", "已读", "以下为新消息",
)


def _is_system_text(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    if any(pattern in value for pattern in _SYSTEM_PATTERNS):
        return True
    # 纯时间/日期分隔
    if value.replace(":", "").replace("：", "").replace("/", "").replace(".", "").isdigit():
        return True
    return False


class Agent:
    """消息从「发现」到「发送」的完整流水线，事件通过队列推给 UI。"""

    def __init__(
        self,
        config: Config,
        platform: Platform,
        dry_run: Optional[bool] = None,
        notify: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.platform = platform
        self.runtime = config.runtime
        self.dry_run = self.runtime.dry_run if dry_run is None else dry_run
        self.notify = notify

        self.persona: Persona = load_persona(config.resolve(config.persona))
        self.memory = MemoryStore(
            config.resolve(config.memory.db_path),
            config.memory.recent_limit,
            config.memory.summarize_after,
        )
        self.tools = ToolBox(
            config.resolve(config.tools.price_file),
            config.resolve(config.tools.leads_file),
            config.tools.enabled,
        )
        self.vision = VisionClient(config.ai)
        self.embeddings = EmbeddingClient(
            config.ai.embedding_model, config.ai.base_url, config.ai.api_key,
            config.resolve(config.rag.embedding_cache),
        )
        self.kb = KnowledgeBase(config.resolve(config.kb_dir), config.rag.mode, self.embeddings)
        self.ecommerce = EcommerceService(config.resolve("data/ecommerce/orders.json"))
        self.router = ModelRouter(config.ai, dry_run=self.dry_run)
        self.router.client.dry_run = not self.router.client.ready

        self.ocr = OcrEngine()
        self.whitelist = Whitelist(config.safety.whitelist, self.runtime.allow_group_chat)
        self.guardrails = Guardrails(
            config.safety.forbidden_keywords,
            config.safety.escalate_keywords,
            config.safety.forbidden_self_disclosure,
            config.safety.max_reply_chars,
        )
        self.limiter = RateLimiter(self.runtime.max_actions_per_min, self.runtime.cooldown_s)
        self.sender = SafeSender(platform, config, self.whitelist, self.ocr)
        saved_layout = False
        settings_path = config.resolve(config.settings_path)
        if os.path.exists(settings_path):
            try:
                import json as _json
                with open(settings_path, "r", encoding="utf-8") as handle:
                    saved_layout = bool((_json.load(handle) or {}).get("layout"))
            except Exception:
                saved_layout = False
        if not saved_layout and hasattr(platform, "layout_preset"):
            try:
                for key, value in platform.layout_preset().items():
                    setattr(config.layout, key, float(value))
            except Exception:
                pass
        self.layout = LayoutCalculator(
            config.layout.nav_ratio,
            config.layout.list_ratio,
            config.layout.header_ratio,
            config.layout.input_ratio,
            config.layout.overrides,
        )
        self.audit = AuditStore(
            config.resolve(config.audit.db_path),
            config.resolve(config.audit.evidence_dir),
            config.audit.save_screenshots,
        )
        self.media_dir = config.resolve("data/media")
        os.makedirs(self.media_dir, exist_ok=True)
        self.avatar_dir = config.resolve("data/avatars")
        os.makedirs(self.avatar_dir, exist_ok=True)

        # ---- 运行状态 ----
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)
        self.paused = threading.Event()
        self._stopped = threading.Event()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._pending_order: List[str] = []
        self._replied: Dict[str, str] = {}
        self._conv: Dict[str, Dict[str, Any]] = {}
        self._instruction: str = ""
        self._summarizing: set = set()
        self._last_capture_error: float = 0.0
        self._deep_collect: Optional[List[str]] = None
        self._view_history: Dict[str, List[Tuple[str, str]]] = {}
        self._last_click: Dict[str, float] = {}
        self._sentinel = PixelSentinel()
        self._peer_positions: Dict[str, Dict[str, float]] = {}
        self._deep_page: Optional[int] = None
        self._last_error: str = ""
        self._reading: set = set()

    # ---------- 运行控制 ----------
    def set_dry_run(self, value: bool) -> None:
        self.dry_run = bool(value)
        self.router.set_dry_run(not self.router.client.ready)
        self.emit("status", message=f"真实发送={'开' if not self.dry_run else '关（仅草稿）'}")

    def set_model(self, model_id: str) -> bool:
        ok = self.router.switch(model_id)
        self.router.client.dry_run = not self.router.client.ready
        self.emit("model", message=f"模型已切换为 {self.router.current}", model=self.router.current, ok=ok)
        return ok

    def current_model(self) -> str:
        return self.router.current

    def model_options(self) -> List[Dict[str, object]]:
        result = []
        for option in self.router.options:
            item = dict(option.as_dict())
            item["has_key"] = bool(option.api_key)
            result.append(item)
        return result

    def layout_ratios(self) -> Dict[str, float]:
        return {
            "nav_ratio": self.config.layout.nav_ratio,
            "list_ratio": self.config.layout.list_ratio,
            "header_ratio": self.config.layout.header_ratio,
            "input_ratio": self.config.layout.input_ratio,
        }

    def capture_preview(self) -> Optional[Tuple[Any, Regions]]:
        image = self.platform.capture()
        if image is None:
            return None
        return image, self.layout.regions(image.width, image.height)

    def apply_settings(self, data: Dict[str, Any]) -> None:
        """控制台保存设置后热更新：Key / 白名单 / 布局 / 视觉与向量模型。"""
        keys = data.get("api_keys") or {}
        if isinstance(keys, dict):
            self.config.ai.model_keys = {str(k): str(v) for k, v in keys.items() if v}
        whitelist = data.get("whitelist")
        if isinstance(whitelist, list):
            self.config.safety.whitelist = [str(item) for item in whitelist if str(item).strip()]
            self.whitelist = Whitelist(self.config.safety.whitelist, self.runtime.allow_group_chat)
        layout = data.get("layout") or {}
        if isinstance(layout, dict):
            for key in ("nav_ratio", "list_ratio", "header_ratio", "input_ratio"):
                if key in layout:
                    try:
                        setattr(self.config.layout, key, float(layout[key]))
                    except (TypeError, ValueError):
                        continue
            self.layout = LayoutCalculator(
                self.config.layout.nav_ratio, self.config.layout.list_ratio,
                self.config.layout.header_ratio, self.config.layout.input_ratio,
                self.config.layout.overrides,
            )
        if isinstance(data.get("vision_model"), str):
            self.config.ai.vision_model = data["vision_model"]
            self.vision = VisionClient(self.config.ai)
        if isinstance(data.get("embedding_model"), str):
            self.config.ai.embedding_model = data["embedding_model"]
            self.embeddings = EmbeddingClient(
                self.config.ai.embedding_model, self.config.ai.base_url, self.config.ai.api_key,
                self.config.resolve(self.config.rag.embedding_cache),
            )
            self.kb = KnowledgeBase(self.config.resolve(self.config.kb_dir),
                                    self.config.rag.mode, self.embeddings)
        # 重建模型路由（保留当前模型）
        current = self.router.current
        self.router = ModelRouter(self.config.ai, dry_run=self.dry_run)
        if not self.router.switch(current):
            self.router.switch(self.config.ai.model)
        self.router.client.dry_run = not self.router.client.ready
        self.emit("settings", message="设置已应用")

    def set_instruction(self, text: str) -> None:
        self._instruction = (text or "").strip()

    def pause(self) -> None:
        self.paused.set()
        self.emit("status", message="已暂停自动回复")

    def resume(self) -> None:
        self.paused.clear()
        self.emit("status", message="已恢复自动回复")

    def stop(self) -> None:
        self._stopped.set()
        self.paused.clear()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    # ---------- 事件 ----------
    def emit(self, kind: str, **fields: Any) -> None:
        event = {"type": kind, "ts": time.time(), **fields}
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass
        summary = fields.get("message") or fields.get("reason") or fields.get("status") or ""
        peer = fields.get("peer", "")
        if kind in ("draft", "sent", "escalate", "error", "tool", "model"):
            self.notify(f"[{kind}] {peer} {summary}".strip())

    # ---------- 主循环 ----------
    def run(self, cycles: Optional[int] = None, once: bool = False) -> None:
        self.emit("status", message=f"启动：platform={self.platform.name} dry_run={self.dry_run} "
                                    f"model={self.router.current} persona={self.persona.display_name} "
                                    f"kb={self.kb.size} 记忆={self.config.memory.db_path}")
        if getattr(self.runtime, "offscreen", False) and hasattr(self.platform, "mount_offscreen"):
            try:
                if self.platform.mount_offscreen():
                    self.emit("status", message="微信已移入离屏空间（PrintWindow 抓图）")
            except Exception:
                pass
        iterations = 1 if once else cycles
        executed = 0
        try:
            while not self.stopped:
                if iterations is not None and executed >= iterations:
                    return
                if self.paused.is_set():
                    now = time.time()
                    if now - getattr(self, "_last_sync", 0) >= 2.0:
                        self._last_sync = now
                        self.sync_conversations()
                    time.sleep(0.2)
                    continue
                self.cycle()
                executed += 1
                if iterations is not None and executed >= iterations:
                    return
                time.sleep(self.runtime.poll_interval_s)
        except KeyboardInterrupt:
            self.emit("status", message="收到 Ctrl+C，退出")
        finally:
            self.platform.close()

    # ---------- 单轮 ----------
    def cycle(self) -> int:
        if getattr(self.platform, "passive", False):
            if not getattr(self, "_passive_notice", False):
                self._passive_notice = True
                self.emit("status", message="只读模式：不点击、不滚动、不发送；请在微信手动切换会话")
            return 0
        if in_quiet_hours(self.runtime.quiet_hours):
            return 0
        image = self.platform.capture()
        if image is None:
            now = time.time()
            if now - self._last_capture_error > 10:
                self._last_capture_error = now
                self.emit("error", message="微信窗口不可用（未运行/已隐藏），已暂停自动扫描；恢复后点「继续」")
                self.pause()
            return 0
        regions = self.layout.regions(image.width, image.height)
        try:
            self.sync_conversations()
        except Exception:
            pass
        if not self._pending and not self._sentinel.should_scan(image, regions):
            return 0
        targets_fn = getattr(self.platform, "unread_targets", None)
        if callable(targets_fn):
            processed = 0
            for _name, x, y in (targets_fn() or [])[: self.runtime.max_unread_per_cycle]:
                badge = type("B", (), {"center": (x, y)})()
                if self._handle_unread(badge, regions):
                    processed += 1
            return processed
        badges = find_red_badges(image, regions)
        if not badges:
            # 当前打开的会话可能显示绿色选中徽标，不产生红点；白名单内也处理
            if self.process_current() is not None:
                return 1
            return 0
        self.emit("cycle", badges=len(badges), dry_run=self.dry_run)

        # 同一行可能检出多个红点（头像徽标 + 免打扰图标），先按 y 合并
        merged = []
        for badge in sorted(badges, key=lambda item: item.center[1]):
            if merged and abs(badge.center[1] - merged[-1].center[1]) <= 40:
                continue
            merged.append(badge)
        badges = merged

        # 有 OCR 时先用会话名预检：只点白名单会话，避免误标已读
        candidates = []
        names = self._session_names(image, regions)
        for badge in badges:
            name = self._name_for_badge(image, regions, badge) or (
                self._match_name(badge, names) if names else None
            )
            if name:
                skip, reason = self._should_skip_by_name(name)
                if skip:
                    self._touch(name, status="已跳过")
                    self.audit.record(peer=name, action="skip", status="precheck_skip", detail=reason)
                    self.emit("skip", peer=name, reason=reason)
                    continue
            elif getattr(self.platform, "strict_name_precheck", False):
                self.audit.record(peer="", action="skip", status="precheck_unknown",
                                  detail="未识别会话名，跳过以免误点")
                self.emit("skip", peer="", reason="未识别会话名（未点开）")
                continue
            if name and time.time() - self._last_click.get(name, 0) < 15:
                self.emit("skip", peer=name, reason="点击冷却中（15s）")
                continue
            if name:
                self._last_click[name] = time.time()
            candidates.append(badge)
        processed = 0
        for badge in candidates[: self.runtime.max_unread_per_cycle]:
            if self._handle_unread(badge, regions):
                processed += 1
        if processed == 0 and self.process_current() is not None:
            processed = 1
        return processed

    # ---------- 会话/历史 ----------
    def _touch(self, peer: str, status: str = "", unread: Optional[int] = None,
               text: str = "", channel: str = "wechat") -> None:
        entry = self._conv.setdefault(
            peer, {"peer": peer, "unread": 0, "last_ts": 0.0, "last_text": "",
                   "status": "", "channel": channel}
        )
        entry.setdefault("channel", channel)
        entry["last_ts"] = time.time()
        if status:
            entry["status"] = status
        if unread is not None:
            entry["unread"] = unread
        if text:
            entry["last_text"] = text

    def mark_read(self, peer: str) -> None:
        """点开会话即视为已读：清掉控制台红点，避免点完还挂 1。"""
        if not peer:
            return
        entry = self._conv.get(peer)
        if entry is not None:
            entry["unread"] = 0

    def conversation(self, peer: str) -> List[Dict[str, str]]:
        snapshot = self._view_history.get(peer)
        if snapshot is not None:
            return [{"dir": direction, "kind": "text", "text": text}
                    for direction, text in snapshot]
        rows = self.memory.recent(peer, 30)
        if rows:
            return [{"dir": direction, "kind": kind, "text": text}
                    for _id, direction, kind, text in rows]
        return [{"dir": "sys", "kind": "text",
                 "text": "（还没有读取到该会话记录：请在微信里打开聊天，然后点顶部「读取当前会话」）"}]

    def _read_chat_page(self, image, regions: Regions):
        """读取当前可见聊天区：返回 [(dir, text)] 列表。"""
        try:
            crop = image.crop(regions.messages)
            small, scale = downscale_for_ocr(crop)
            lines = self.ocr.recognize(small)
        except Exception:
            return []
        offset_x, offset_y = regions.messages[0], regions.messages[1]
        for line in lines:
            x1, y1, x2, y2 = line.bbox
            line.bbox = (x1 / scale + offset_x, y1 / scale + offset_y,
                         x2 / scale + offset_x, y2 / scale + offset_y)
        lines = [line for line in lines if not _is_system_text(line.text)]
        media_items = detect_chat_images(image, regions, self.media_dir)
        image_boxes = [item.bbox for item in media_items if item.kind == "image" and item.bbox]

        def inside_image(line) -> bool:
            cx, cy = line.center
            return any(x1 - 4 <= cx <= x2 + 4 and y1 - 4 <= cy <= y2 + 4
                       for x1, y1, x2, y2 in image_boxes)

        lines = [line for line in lines if not inside_image(line)]
        chat_lines = lines_from_ocr_colored(image, regions, lines, scale=1.0)
        messages = []
        for line in chat_lines:
            text = line.text.strip()
            if text:
                messages.append((line.direction, text))
        for _item in media_items:
            messages.append(("in", "[图片]"))
        return messages

    def select_conversation(self, peer: str) -> bool:
        """点开指定会话并读取聊天记录（可见列表优先）。"""
        self._reading.add(peer)
        self._view_history[peer] = [("sys", "正在读取聊天记录…")]
        try:
            return self._select_conversation_inner(peer)
        finally:
            self._reading.discard(peer)

    def _select_conversation_inner(self, peer: str) -> bool:
        if not self.open_conversation(peer):
            self._view_history[peer] = [
                ("in", "（无法自动打开该会话：请在微信里手动打开，然后点「读取当前会话」）")
            ]
            self._touch(peer, status="打开失败")
            return False
        return self.read_conversation(peer, open_first=False)

    def simulate_incoming(self, peer: str, text: str, auto: bool = True) -> Optional[Dict[str, Any]]:
        """项目内模拟客户来消息：走完整思考/回复链路，绝不操作真实微信。"""
        text = (text or "").strip()
        if not text:
            return None
        peer = (peer or "模拟客户").strip()
        if not peer.startswith("【模拟】"):
            peer = "【模拟】" + peer
        history = self.memory.history_lines(peer, self.config.memory.recent_limit)
        profile = self.memory.profile(peer)
        decision = self.guardrails.assess_incoming(text)
        self.memory.add(peer, "in", "text", text)
        view = self._view_history.setdefault(peer, [])
        view.append(("in", text))
        if decision.action == "escalate":
            label = f"（转人工）{decision.reason}"
            view.append(("out", label))
            self.memory.add(peer, "out", "text", label)
            self._touch(peer, status="转人工", unread=0, text=label)
            self.emit("simulate", peer=peer, incoming=text, reply=label, auto=True,
                      model="guardrail", latency_ms=0, tools=[], escalated=True)
            return {"peer": peer, "incoming": text, "reply": label, "auto": True,
                    "model": "guardrail", "latency_ms": 0, "tools": [],
                    "escalated": True}
        tool_results = self.tools.prefetch(text)
        for result in tool_results:
            self.emit("tool", name=result.name, summary=result.summary, ok=result.ok,
                      simulated=True)
        reply, model, latency = self._generate(
            [text], text, history, profile, tool_results, is_group_name(peer),
            peer=peer
        )
        if reply is None:  # 偶发网络/限流，重试一次
            reply, model, latency = self._generate(
                [text], text, history, profile, tool_results, is_group_name(peer)
            )
        if reply is None:
            return None
        if auto:
            view.append(("out", reply))
            self.memory.add(peer, "out", "text", reply)
            status = "模拟已回复"
        else:
            self._pending[peer] = {
                "peer": peer, "reply": reply, "incoming": text, "image": None,
                "regions": None, "model": model, "latency_ms": latency,
                "incoming_items": [text], "history": history, "profile": profile,
                "tool_results": [r.as_dict() for r in tool_results],
                "is_group": is_group_name(peer), "simulated": True,
            }
            if peer in self._pending_order:
                self._pending_order.remove(peer)
            self._pending_order.append(peer)
            status = "模拟草稿"
        self._touch(peer, status=status, unread=0, text=reply)
        self._conv[peer]["simulated"] = True
        self.audit.record(peer=peer, action="simulate", status=status,
                          incoming=text, reply=reply, model=model, latency_ms=latency,
                          detail="项目内模拟，未操作真实微信")
        self.emit("simulate", peer=peer, incoming=text, reply=reply, auto=auto,
                  model=model, latency_ms=latency,
                  tools=[r.name for r in tool_results])
        return {"peer": peer, "incoming": text, "reply": reply, "auto": auto,
                "model": model, "latency_ms": latency,
                "tools": [r.summary for r in tool_results]}

    def handle_inbound(self, peer: str, text: str, auto: bool = True,
                       context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """外部 IM / 电商平台入站：走同一套人设/知识库/工具/护栏。"""
        text = (text or "").strip()
        if not text:
            return None
        peer = (peer or "unknown").strip()
        channel = peer.split(":", 1)[0] if ":" in peer else "wechat"
        ecom_context = ""
        if channel in ("taobao", "jd", "pdd", "douyin_shop", "ecommerce"):
            ecom_context = self.ecommerce.describe(peer, text, context or {})
        history = self.memory.history_lines(peer, self.config.memory.recent_limit)
        profile = self.memory.profile(peer)
        decision = self.guardrails.assess_incoming(text)
        self.memory.add(peer, "in", "text", text)
        view = self._view_history.setdefault(peer, [])
        view.append(("in", text))
        if decision.action == "escalate":
            reply = f"（转人工）{decision.reason}"
            view.append(("out", reply))
            self.memory.add(peer, "out", "text", reply)
            self._touch(peer, status="转人工", unread=0, text=reply, channel=channel)
            self.emit("inbound", peer=peer, incoming=text, reply=reply, escalated=True)
            return {"peer": peer, "incoming": text, "reply": reply, "escalated": True}
        tool_results = self.tools.prefetch(text)
        reply, model, latency = self._generate(
            [text], text, history, profile, tool_results, is_group_name(peer),
            peer=peer, context=ecom_context
        )
        if reply is None:
            return None
        self._touch(peer, status="外部消息已回复" if auto else "外部消息草稿",
                    unread=0, text=reply, channel=channel)
        if auto:
            view.append(("out", reply))
            self.memory.add(peer, "out", "text", reply)
        else:
            self._pending[peer] = {"peer": peer, "reply": reply, "incoming": text,
                                   "image": None, "regions": None, "model": model,
                                   "latency_ms": latency, "incoming_items": [text],
                                   "history": history, "profile": profile,
                                   "tool_results": [r.as_dict() for r in tool_results],
                                   "is_group": is_group_name(peer), "external": True}
            if peer in self._pending_order:
                self._pending_order.remove(peer)
            self._pending_order.append(peer)
        self.audit.record(peer=peer, action="inbound", status="已回复" if auto else "草稿",
                          incoming=text, reply=reply, model=model, latency_ms=latency,
                          detail="外部 IM 入站")
        self.emit("inbound", peer=peer, incoming=text, reply=reply, auto=auto,
                  model=model, latency_ms=latency)
        return {"peer": peer, "incoming": text, "reply": reply, "auto": auto,
                "model": model, "latency_ms": latency}

    def read_current(self) -> Optional[str]:
        """读取微信当前已经打开的会话（最可靠，不尝试切换会话）。"""
        image = self.platform.capture()
        if image is None:
            return None
        regions = self.layout.regions(image.width, image.height)
        peer = self.sender.read_header(image, regions)
        if not peer:
            return None
        if self.read_conversation(peer, open_first=False):
            return peer
        return None

    def read_conversation(self, peer: str, open_first: bool = True) -> bool:
        """读取聊天历史；open_first=False 表示只读当前打开会话。"""
        if open_first and not self.open_conversation(peer):
            self.emit("error", message=f"打开会话失败：{peer}")
            self._view_history[peer] = [("in", "（无法自动打开该会话，请在微信中手动打开后点「读取当前会话」）")]
            return False
        collected: List[Tuple[str, str]] = []
        seen = set()
        try:
            image = self.platform.capture()
            if image is None:
                return False
            regions = self.layout.regions(image.width, image.height)
            # 先滚到底
            if callable(getattr(self.platform, "scroll", None)):
                x = regions.chat[0] + (regions.chat[2] - regions.chat[0]) * 0.5
                y = regions.messages[1] + (regions.messages[3] - regions.messages[1]) * 0.5
                self.platform.scroll(x, y, -8, 10)
                time.sleep(0.4)
            for page in range(3):
                image = self.platform.capture()
                if image is None:
                    break
                regions = self.layout.regions(image.width, image.height)
                page_messages = self._read_chat_page(image, regions)
                new_count = 0
                for direction, text in page_messages:
                    key = (direction, text)
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.insert(0, (direction, text))
                    new_count += 1
                if page == 0 and collected:
                    # 首屏先显示，别让用户干等
                    self._view_history[peer] = list(reversed(collected))[-80:]
                    self.emit("read", peer=peer, count=len(collected), partial=True)
                if page >= 1 and new_count == 0:
                    break
                if callable(getattr(self.platform, "scroll", None)):
                    x = regions.chat[0] + (regions.chat[2] - regions.chat[0]) * 0.5
                    y = regions.messages[1] + (regions.messages[3] - regions.messages[1]) * 0.5
                    self.platform.scroll(x, y, 6, 1)  # 向上翻一页
                    time.sleep(0.3)
        except Exception as exc:
            self.emit("error", message=f"读取聊天记录失败：{exc}")
        if not collected:
            collected = [("in", "（未读取到聊天记录：可能为空会话、纯图片或 OCR 失败）")]
        self._view_history[peer] = collected[-80:]
        self._touch(peer, status="已读取", unread=0,
                    text=collected[-1][1] if collected else "")
        self.emit("read", peer=peer, count=len(collected))
        return True

    def conversations(self) -> List[Dict[str, Any]]:
        return sorted(self._conv.values(),
                      key=lambda item: (item.get("order", 10**9), -item["last_ts"]))

    @property
    def pending(self) -> Optional[Dict[str, Any]]:
        if not self._pending_order:
            return None
        return self._pending.get(self._pending_order[-1])

    def pending_for(self, peer: str) -> Optional[Dict[str, Any]]:
        return self._pending.get(peer)

    # ---------- 会话同步 ----------
    def _scroll_session_list(self, delta: int, times: int = 1) -> None:
        scroll = getattr(self.platform, "scroll", None)
        if not callable(scroll):
            return
        image = self.platform.capture()
        if image is None:
            return
        regions = self.layout.regions(image.width, image.height)
        left, _top, right, _bottom = regions.session_list
        x = left + (right - left) * 0.5
        y = image.height * 0.45
        try:
            scroll(x, y, delta, times)
        except Exception:
            pass
        time.sleep(0.3)

    def _session_names(self, image, regions: Regions) -> List[Tuple[str, float]]:
        if not self.ocr.available:
            return []
        try:
            lines = self.ocr.recognize(image.crop(regions.session_list))
        except Exception:
            return []
        names = []
        for line in lines:
            text = (line.text or "").strip()
            if not text or line.score < 0.35:
                continue
            height = line.bbox[3] - line.bbox[1]
            if height < 20 or len(text) > 30:
                continue
            if any(token in text for token in ("…", "：", "［", "］", "[", "]", "条]", "拍了拍",
                                                "..", "．．", "。。", "⋯", "搜索")):
                continue
            if text.replace(":", "").replace("：", "").isdigit():
                continue
            names.append((text, (line.bbox[1] + line.bbox[3]) / 2.0))
        return names

    @staticmethod
    def _clean_peer_name(name: str) -> str:
        import re as _re
        value = (name or "").strip()
        value = _re.sub(r"[\s·.．…⋯/]+$", "", value)
        return _re.sub(r"\s+", "", value).lower()

    def _canonical_peer(self, name: str) -> str:
        import difflib
        cleaned = self._clean_peer_name(name)
        if not cleaned:
            return name
        best, best_ratio = None, 0.0
        for existing in self._conv.keys():
            other = self._clean_peer_name(existing)
            if not other:
                continue
            if other == cleaned:
                return existing
            ratio = difflib.SequenceMatcher(None, other, cleaned).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, existing
        if best and best_ratio >= 0.86:
            return best
        return name

    @staticmethod
    def _detect_avatar(image, regions: Regions, name_x: float, row_y: float, row_h: float):
        """在名字左侧区域找出最大的近似方形连通块，作为头像框。"""
        import numpy as _np
        from collections import deque

        session_left = int(regions.session_list[0])
        session_top = int(regions.session_list[1])
        left = session_left + 6
        right = int(min(session_left + name_x - 4, left + 118))
        top = int(max(0, session_top + row_y - row_h * 2.4))
        bottom = int(min(image.height, session_top + row_y + row_h * 2.4))
        if right - left < 30 or bottom - top < 40:
            return None
        band = image.crop((left, top, right, bottom)).convert("RGB")
        array = _np.array(band, dtype=_np.int16)
        border = _np.concatenate([
            array[0:2].reshape(-1, 3), array[-2:].reshape(-1, 3),
            array[:, 0:2].reshape(-1, 3), array[:, -2:].reshape(-1, 3),
        ])
        background = _np.median(border, axis=0)
        mask = _np.abs(array - background).sum(axis=2) > 34
        red = (array[:, :, 0] >= 140) & (array[:, :, 0] - array[:, :, 1] >= 30) \
            & (array[:, :, 0] - array[:, :, 2] >= 30)
        mask &= ~red
        height, width = mask.shape
        seen = _np.zeros_like(mask, dtype=bool)
        best = None
        for sy in range(height):
            if not mask[sy].any():
                continue
            for sx in _np.where(mask[sy] & ~seen[sy])[0]:
                if seen[sy, sx]:
                    continue
                queue = deque([(int(sy), int(sx))])
                seen[sy, sx] = True
                min_x = max_x = int(sx)
                min_y = max_y = int(sy)
                pixels = 0
                while queue:
                    cy, cx = queue.popleft()
                    pixels += 1
                    if cx < min_x: min_x = cx
                    if cx > max_x: max_x = cx
                    if cy < min_y: min_y = cy
                    if cy > max_y: max_y = cy
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
                w = max_x - min_x + 1
                h = max_y - min_y + 1
                if pixels < 350 or w < 36 or h < 36 or w > 130 or h > 130:
                    continue
                aspect = w / float(h)
                if aspect < 0.65 or aspect > 1.55:
                    continue
                score = pixels
                if best is None or score > best[0]:
                    best = (score, min_x, min_y, max_x, max_y)
        if best is None:
            return None
        _score, x1, y1, x2, y2 = best
        pad = 3
        gx1 = max(0, left + x1 - pad)
        gy1 = max(0, top + y1 - pad)
        size = max(x2 - x1 + 1, y2 - y1 + 1) + pad * 2
        piece = image.crop((gx1, gy1, min(image.width, gx1 + size), min(image.height, gy1 + size)))
        if float(_np.array(piece.convert("RGB")).std()) < 4.0:
            return None
        return piece

    def _ensure_avatar(self, name: str) -> Optional[str]:
        import hashlib as _hash
        path = os.path.join(self.avatar_dir, _hash.sha1(name.encode("utf-8")).hexdigest() + ".png")
        if os.path.exists(path):
            return path
        try:
            from PIL import ImageDraw
            digest = _hash.md5(name.encode("utf-8")).hexdigest()
            color = (int(digest[0:2], 16) % 160 + 80,
                     int(digest[2:4], 16) % 160 + 80,
                     int(digest[4:6], 16) % 160 + 80)
            image = Image.new("RGB", (64, 64), color)
            draw = ImageDraw.Draw(image)
            draw.text((24, 22), (name.strip()[:1] or "?"), fill=(255, 255, 255))
            image.save(path)
            return path
        except Exception:
            return None

    def _apply_order(self, names: List[str]) -> None:
        ordered = []
        for name in names:
            if name in self._conv and name not in ordered:
                ordered.append(name)
        rest = sorted([name for name in self._conv if name not in ordered],
                      key=lambda name: self._conv[name].get("order", 10**9))
        for index, name in enumerate(ordered + rest):
            self._conv[name]["order"] = index

    def _name_for_badge(self, image, regions: Regions, badge) -> Optional[str]:
        if not self.ocr.available:
            return None
        left = regions.session_list[0] + 70
        right = regions.session_list[2] - 10
        top = max(0, int(badge.center[1]) - 70)
        bottom = min(image.height, int(badge.center[1]) + 35)
        if right - left < 40 or bottom - top < 30:
            return None
        try:
            crop = image.crop((left, top, right, bottom))
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
            lines = self.ocr.recognize(crop)
        except Exception:
            return None
        best = None
        for line in lines:
            text = (line.text or "").strip()
            if line.score < 0.35 or not text:
                continue
            height = line.bbox[3] - line.bbox[1]
            if height < 16:
                continue
            if any(token in text for token in ("…", "：", "[", "]", "条]", "搜索", "拍了拍")):
                continue
            if text.replace(":", "").replace("：", "").isdigit():
                continue
            if best is None or height > best[0]:
                best = (height, text)
        return best[1] if best else None

    def _match_name(self, badge, names: List[Tuple[str, float]]) -> Optional[str]:
        if not names:
            return None
        _cx, cy = badge.center
        best = min(names, key=lambda item: abs(item[1] - cy))
        return best[0] if abs(best[1] - cy) <= 90 else None

    def _should_skip_by_name(self, name: str) -> Tuple[bool, str]:
        if is_group_name(name):
            if not self.runtime.allow_group_chat:
                return True, "群聊自动回复未开启（未点开）"
            if self.runtime.group_whitelist:
                normalized = normalize_name(name)
                matched = any(normalize_name(item) in normalized or normalized in normalize_name(item)
                              for item in self.runtime.group_whitelist if item.strip())
                if not matched:
                    return True, "群不在群白名单（未点开）"
            return False, ""
        allowed, _matched = self.whitelist.matches(name)
        return (False, "") if allowed else (True, "不在白名单（未点开）")

    def _read_items(self, image, regions: Regions) -> List[MessageItem]:
        structured = self.platform.read_incoming_items()
        if structured is not None:
            items = list(structured)
        else:
            if not self.ocr.available:
                self.emit("error", message="OCR 不可用")
                return []
            try:
                lines = self.ocr.recognize(image.crop(regions.messages))
            except Exception as exc:
                self.emit("error", message=f"OCR 失败: {exc}")
                return []
            offset_x, offset_y = regions.messages[0], regions.messages[1]
            for line in lines:
                x1, y1, x2, y2 = line.bbox
                line.bbox = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            lines = [line for line in lines if not _is_system_text(line.text)]
            chat_lines = lines_from_ocr(lines, regions)
            last_out = -1
            for index, line in enumerate(chat_lines):
                if line.direction == "out":
                    last_out = index
            items = [MessageItem(kind="text", text=line.text)
                     for line in chat_lines[last_out + 1:] if line.direction == "in"]
            media_items = extract_items(image, regions, lines, self.media_dir,
                                        detect_media=self.vision.available)
            items.extend(item for item in media_items if item.kind != "text")
        for item in items:
            if item.kind == "image":
                if self.vision.available and item.media_path:
                    item.description = self.vision.describe(item.media_path)
                elif not item.description:
                    item.description = "（未启用视觉模型，无法读图）"
                self.emit("media", media_kind=item.kind, description=item.description,
                          media_path=item.media_path)
        return items

    def sync_conversations(self, deep: bool = False) -> int:
        snapshot = getattr(self.platform, "session_snapshot", None)
        if callable(snapshot):
            rows = snapshot() or []
            order_names = []
            for row in rows:
                name = str(row.get("name") or "")
                if not name:
                    continue
                entry = self._conv.get(name)
                if entry is None:
                    self._touch(name, status="未读" if row.get("unread") else "空闲",
                                text=str(row.get("preview") or ""))
                    entry = self._conv[name]
                else:
                    entry["last_text"] = str(row.get("preview") or entry.get("last_text", ""))
                entry["unread"] = int(row.get("unread") or 0)
                entry["time"] = str(row.get("time") or "")
                avatar = self._ensure_avatar(name)
                if avatar:
                    entry["avatar_path"] = avatar
                order_names.append(name)
            self._apply_order(order_names)
            return len(rows)

        if not self.ocr.available:
            return 0
        scroll = getattr(self.platform, "scroll", None)
        if deep and callable(scroll):
            self._scroll_session_list(8, 14)
            collected: List[str] = []
            self._deep_collect = collected
            try:
                for index in range(20):
                    before = len(collected)
                    self._deep_page = index
                    self.sync_conversations(False)
                    self._scroll_session_list(-6, 1)
                    if index >= 1 and len(collected) == before:
                        break
            finally:
                self._deep_collect = None
                self._deep_page = None
            self._scroll_session_list(8, 12)
            ordered = [name for name in dict.fromkeys(collected)]
            for index, name in enumerate(ordered):
                if name in self._conv:
                    self._conv[name]["order"] = index
            return len(ordered)

        if self.platform.find_window() is None:
            return 0
        image = self.platform.capture()
        if image is None:
            return 0
        regions = self.layout.regions(image.width, image.height)
        try:
            lines = self.ocr.recognize(image.crop(regions.session_list))
        except Exception:
            return 0
        if not lines:
            return 0
        import re as _re
        import numpy as np
        time_re = _re.compile(r"^(\d{1,2}[:：]\d{2}|\d{1,2}/\d{1,2}|昨天|前天|星期[一二三四五六日天]|上午|下午|中午|晚上).*$")

        def is_time(text: str) -> bool:
            return bool(time_re.match((text or "").strip().rstrip(".")))

        def is_bad_name(text: str) -> bool:
            value = (text or "").strip()
            if not value or len(value) > 30:
                return True
            if value in ("搜索", "A", "AI"):
                return True
            if any(token in value for token in ("…", "：", "［", "］", "[", "]", "条]", "拍了拍",
                                                "..", "．．", "。。", "⋯")):
                return True
            if _re.search(r"\d{6,}", value):
                return True
            return value.replace(":", "").replace("：", "").isdigit()

        area = np.array(image.crop(regions.session_list).convert("RGB"))
        red = (area[:, :, 0].astype(np.int16) >= 170) & (area[:, :, 1].astype(np.int16) <= 110) \
            & (area[:, :, 2].astype(np.int16) <= 110)
        times = [ln for ln in lines if is_time(ln.text)]
        parsed = []
        for anchor in times:
            anchor_y = (anchor.bbox[1] + anchor.bbox[3]) / 2.0
            candidates = [ln for ln in lines
                          if not is_time(ln.text) and ln.bbox[0] < 320
                          and not is_bad_name(ln.text)
                          and anchor_y - 30 <= (ln.bbox[1] + ln.bbox[3]) / 2.0 <= anchor_y + 22]
            if not candidates:
                continue
            name_line = max(candidates, key=lambda ln: (ln.bbox[3] - ln.bbox[1], ln.score))
            name = name_line.text.strip()
            if name.endswith("禁") and len(name) > 2 and "群" in name:
                original = name
                name = name[:-1]
                self._conv.pop(original, None)
            name = self._canonical_peer(name)
            if any(item["name"] == name for item in parsed):
                continue
            preview = ""
            later = [t for t in times if (t.bbox[1] + t.bbox[3]) / 2.0 > anchor_y + 10]
            next_y = (min((t.bbox[1] + t.bbox[3]) / 2.0 for t in later) - 8) if later else anchor_y + 110
            for ln in sorted(lines, key=lambda l: l.bbox[1]):
                if ln is name_line or is_time(ln.text):
                    continue
                ly = (ln.bbox[1] + ln.bbox[3]) / 2.0
                if name_line.bbox[3] + 1 <= ly <= next_y and ln.bbox[0] < 320:
                    preview = (ln.text or "").strip()
                    break
            name_x = float(name_line.bbox[0])
            name_y1 = int(name_line.bbox[1])
            avatar_left = int(name_x - 87)
            bx1, bx2 = max(0, avatar_left + 42), min(area.shape[1], avatar_left + 92)
            by1, by2 = max(0, name_y1 - 30), min(area.shape[0], name_y1 + 10)
            unread = 0
            if bx2 > bx1 and by2 > by1:
                patch = red[by1:by2, bx1:bx2]
                if int(patch.sum()) >= 20:
                    rows_idx, cols_idx = np.where(patch)
                    height = int(rows_idx.max() - rows_idx.min() + 1)
                    width = int(cols_idx.max() - cols_idx.min() + 1)
                    if 8 <= width <= 30 and 8 <= height <= 30:
                        unread = 1
                        digits = ""
                        try:
                            badge = area[by1 + rows_idx.min():by1 + rows_idx.max() + 1,
                                         bx1 + cols_idx.min():bx1 + cols_idx.max() + 1]
                            big = Image.fromarray(badge).resize(
                                (badge.shape[1] * 4, badge.shape[0] * 4), Image.LANCZOS)
                            texts = "".join(ln.text for ln in self.ocr.recognize(big))
                            digits = "".join(ch for ch in texts if ch.isdigit())
                        except Exception:
                            digits = ""
                        if digits:
                            unread = int(digits[:3])
            parsed.append({"name": name, "preview": preview, "unread": unread,
                           "time": anchor.text.strip().rstrip("."),
                           "name_x": name_x, "anchor_y": anchor_y,
                           "row_y": float(name_line.bbox[1]),
                           "name_h": float(name_line.bbox[3] - name_line.bbox[1])})

        parsed.sort(key=lambda it: float(it.get("anchor_y") or 0))
        for index, item in enumerate(parsed):
            if index + 1 < len(parsed):
                delta = float(parsed[index + 1].get("anchor_y") or 0) - float(item.get("anchor_y") or 0)
            else:
                delta = 92.0
            item["row_height"] = min(120.0, max(56.0, delta))

        for item in parsed:
            name = item["name"]
            entry = self._conv.get(name)
            if entry is None:
                self._touch(name, status="未读" if item["unread"] else "空闲",
                            text=item["preview"] or None)
                entry = self._conv[name]
            else:
                entry["last_text"] = item["preview"] or entry.get("last_text", "")
            entry["unread"] = item["unread"]
            entry["time"] = item["time"]
            name_h = float(item.get("name_h") or 32.0)
            name_top = float(item.get("row_y") or 0)
            name_x_global = regions.session_list[0] + float(item.get("name_x") or 0)
            size = 80
            # 头像在名字左侧约 2.9 倍行高处，按名字位置反推，避免布局 nav 比例错误
            left = int(max(regions.session_list[0] + 6,
                           name_x_global - name_h * 2.9))
            best_avatar, best_std = None, 0.0
            for shift in (0, -6, 6, -12, 12):
                top = int(max(0, name_top - 6 + shift))
                candidate = image.crop((left, top, left + size, top + size))
                std = float(np.array(candidate.convert("RGB")).std())
                if std > best_std:
                    best_std, best_avatar = std, candidate
            if best_avatar is not None and best_std >= 3.0:
                path = os.path.join(self.avatar_dir, __import__("hashlib").sha1(
                    name.encode("utf-8")).hexdigest() + ".png")
                best_avatar.save(path)
                entry["avatar_path"] = path
            else:
                generated = self._ensure_avatar(name)
                if generated:
                    entry["avatar_path"] = generated
        for item in parsed:
            self._peer_positions[item["name"]] = {
                "page": float(self._deep_page or 0),
                "y": float(item.get("row_y", item.get("anchor_y", 0))),
            }
        page_names = [item["name"] for item in parsed if item.get("name")]
        if self._deep_collect is not None:
            self._deep_collect.extend(page_names)
        else:
            self._apply_order(page_names)
        for entry in self._conv.values():
            if not entry.get("avatar_path"):
                generated = self._ensure_avatar(entry["peer"])
                if generated:
                    entry["avatar_path"] = generated
        valid = {item["name"] for item in parsed}
        for name in list(self._conv.keys()):
            if name in valid:
                continue
            if any(token in name for token in ("［", "］", "条]", "…", "：")):
                self._conv.pop(name, None)
        return len(parsed)

    # ---------- 处理一个未读 ----------
    def _handle_unread(self, badge, regions: Regions) -> bool:
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        x, y = click_point(badge, regions)
        self.platform.click(x, y)
        image = settle_capture(
            self.platform.capture, regions,
            self.runtime.settle_s, self.runtime.aggregate_max_wait_s,
        )
        if image is None:
            return False

        header = self.sender.read_header(image, regions)
        peer = header or ""
        is_group = is_group_name(peer)

        items = self._read_items(image, regions)
        if not items:
            self.emit("skip", peer=peer, reason="没有新消息")
            return False

        if is_group:
            allowed, reason = group_allows(
                peer, items,
                self.runtime.allow_group_chat,
                self.runtime.group_whitelist,
                self.runtime.group_reply_mode,
                self.runtime.group_keywords,
            )
            if not allowed:
                self._touch(peer, status="已跳过", unread=0)
                self.audit.record(peer=peer, action="skip", status="group_skipped", detail=reason)
                self.emit("skip", peer=peer, reason=reason)
                return False
            self.emit("group", peer=peer, reason=f"群聊放行（{self.runtime.group_reply_mode}）")
        else:
            allowed, _matched = self.whitelist.matches(peer)
            if not allowed:
                self._touch(peer, status="已跳过", unread=0)
                self.audit.record(peer=peer, action="skip", status="not_whitelisted",
                                  detail="不在白名单")
                self.emit("skip", peer=peer, reason="不在白名单")
                return False

        self._touch(peer, status="处理中", unread=0)
        incoming = [item.prompt_text() for item in items]
        joined = " / ".join(incoming)

        history = self.memory.history_lines(peer, self.config.memory.recent_limit)
        profile = self.memory.profile(peer)
        for item in items:
            self.memory.add(peer, "in", item.kind, item.prompt_text())
        self.emit("incoming", peer=peer, messages=incoming, kinds=[item.kind for item in items])

        signature = normalize_name(joined)
        if self._replied.get(peer) == signature:
            self.emit("skip", peer=peer, reason="重复消息")
            return False

        decision = self.guardrails.assess_incoming(joined)
        if decision.action == "escalate":
            self._touch(peer, status="转人工")
            self.audit.record(peer=peer, action="escalate", status="handoff",
                              incoming=joined, detail=decision.reason)
            self.emit("escalate", peer=peer, reason=decision.reason)
            return False

        if not self.limiter.allowed(peer):
            wait = self.limiter.wait_seconds(peer)
            self.emit("rate_limited", peer=peer, wait=round(wait, 1))
            time.sleep(min(wait, 3.0))
            return False

        tool_results = self.tools.prefetch(joined)
        for result in tool_results:
            self.emit("tool", name=result.name, summary=result.summary, ok=result.ok)

        reply, model, latency = self._generate(incoming, joined, history, profile,
                                               tool_results, is_group, peer=peer)
        if reply is None:
            return False

        self._pending[peer] = {
            "peer": peer, "reply": reply, "incoming": joined, "image": image,
            "regions": regions, "model": model, "latency_ms": latency,
            "incoming_items": incoming, "history": history, "profile": profile,
            "tool_results": [result.as_dict() for result in tool_results], "is_group": is_group,
        }
        if peer in self._pending_order:
            self._pending_order.remove(peer)
        self._pending_order.append(peer)
        self._touch(peer, status="草稿待发", unread=0, text=reply)
        audit_status = "dry_run" if self.dry_run else "draft"
        self.audit.record(peer=peer, action="draft", status=audit_status,
                          incoming=joined, reply=reply, model=model,
                          latency_ms=latency, image=image)
        self.emit("draft", peer=peer, reply=reply, model=model,
                  latency_ms=latency, dry_run=self.dry_run,
                  tools=[result.name for result in tool_results])

        if self.dry_run:
            self.memory.add(peer, "out", "text", reply)
            self._replied[peer] = signature
            self.limiter.record(peer, cost=1)
            self._maybe_summarize(peer)
            return True
        outcome = self.sender.send(peer, reply, image, regions)
        return self._finish_send(peer, reply, outcome, joined, model, latency, signature)

    # ---------- 生成 ----------
    def _friend_persona(self, peer: str) -> str:
        table = self.config.friend_personas or {}
        return (table.get(peer) or table.get("默认") or "").strip()

    def _generate(self, incoming: List[str], joined: str, history: List[str],
                  profile: Dict[str, str], tool_results: List[Any], is_group: bool,
                  peer: str = "", context: str = "") -> Tuple[Optional[str], str, int]:
        chunks = self.kb.search(joined, top_k=self.config.rag.top_k)
        system = build_system_prompt(self.persona, chunks, profile)
        if profile.get("summary"):
            system += f"\n【历史摘要】{profile['summary']}"
        if tool_results:
            lines = "；".join(result.summary for result in tool_results)
            system += f"\n【工具查询结果（可直接引用，不要编造）】{lines}"
        if is_group:
            system += "\n当前是群聊：回复要简短，必要时点名对方，不要刷屏。"
        friend_persona = self._friend_persona(peer)
        if friend_persona:
            system += f"\n【该好友专属人设】{friend_persona}"
        if context:
            system += f"\n{context}"
        if self._instruction:
            system += f"\n【额外要求】{self._instruction}"

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for example in self.persona.examples or []:
            if example.get("user") and example.get("assistant"):
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})
        for line in history:
            role = "assistant" if line.startswith("我方") else "user"
            content = line
            for prefix in ("我方：", "对方：", "我方", "对方"):
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": "\n".join(incoming)})

        schemas = self.tools.schemas() if (self.tools.enabled and self.router.client.ready) else None
        result = self.router.client.chat(
            messages, tools=schemas, executor=self.tools.execute if schemas else None
        )
        if self.router.client.dry_run and tool_results:
            text = "（dry-run）根据工具查询：" + "；".join(r.summary for r in tool_results) + \
                   "。配置 api_key 后由模型润色成自然回复。"
            checked = self.guardrails.check_reply(text)
            if checked.action == "reply":
                return checked.text, "dry-run", result.latency_ms
        if result.error or not result.text.strip():
            if self.router.client.dry_run and tool_results:
                text = "（dry-run）根据工具查询：" + "；".join(r.summary for r in tool_results) + \
                       "。配置 api_key 后由模型润色成自然回复。"
                checked = self.guardrails.check_reply(text)
                return (checked.text if checked.action == "reply" else None), "dry-run", result.latency_ms
            self.audit.record(peer="", action="error", status="llm_failed",
                              incoming=joined, detail=result.error or "空回复",
                              model=result.model, latency_ms=result.latency_ms)
            self._last_error = f"模型调用失败：{result.error or '空回复'}"
            self.emit("error", message=self._last_error)
            return None, result.model, result.latency_ms
        checked = self.guardrails.check_reply(result.text)
        if checked.action != "reply":
            self.audit.record(peer="", action="block", status=checked.action,
                              incoming=joined, reply=result.text, detail=checked.reason,
                              model=result.model, latency_ms=result.latency_ms)
            self._last_error = f"回复被护栏拦截：{checked.reason}"
            self.emit("blocked", reason=checked.reason, reply=result.text)
            return None, result.model, result.latency_ms
        return checked.text, result.model, result.latency_ms

    # ---------- 发送结果 ----------
    def _finish_send(self, peer: str, reply: str, outcome, joined: str,
                     model: str, latency: int, signature: str) -> bool:
        self.audit.record(peer=peer, action="send", status=outcome.outcome.status,
                          incoming=joined, reply=reply, detail=outcome.outcome.detail,
                          model=model, latency_ms=latency)
        if outcome.outcome.status == "verified":
            self.memory.add(peer, "out", "text", reply)
            self._touch(peer, status="已发送", unread=0, text=reply)
            self._replied[peer] = signature
            self.limiter.record(peer, cost=1)
            self._pending.pop(peer, None)
            if peer in self._pending_order:
                self._pending_order.remove(peer)
            self.emit("sent", peer=peer, status="verified", reply=reply,
                      detail=outcome.outcome.detail)
            self._maybe_summarize(peer)
            return True
        self._touch(peer, status="发送未确认")
        self.emit("sent", peer=peer, status=outcome.outcome.status, reply=reply,
                  detail=outcome.outcome.detail)
        return False

    # ---------- 摘要（长记忆压缩） ----------
    def _maybe_summarize(self, peer: str) -> None:
        if peer in self._summarizing or not self.memory.needs_summary(peer):
            return
        if self.router.client.dry_run or not self.router.client.ready:
            return
        self._summarizing.add(peer)
        try:
            rows = self.memory.pending_messages(peer)
            if not rows:
                return
            text = "\n".join(f"{'对方' if d == 'in' else '我方'}：{t}" for _i, d, _k, t in rows)
            summary_prompt = [
                {"role": "system", "content": "把下面的聊天压缩成 3 句话以内的客户档案摘要，保留需求、报价、交期、称呼。"},
                {"role": "user", "content": text[:4000]},
            ]
            result = self.router.client.chat(summary_prompt, retries=0)
            if result.text and not result.error:
                self.memory.set_summary(peer, result.text.strip(), self.memory.last_id(peer))
                self.emit("memory", peer=peer, message="已更新会话摘要", summary=result.text.strip()[:120])
        finally:
            self._summarizing.discard(peer)

    @staticmethod
    def _same_peer(current: str, peer: str) -> bool:
        left, right = normalize_name(current), normalize_name(peer)
        if not left or not right:
            return False
        if left == right or left in right or right in left:
            return True
        head = min(4, len(right))
        return left.startswith(right[:head]) or right.startswith(left[:head])

    def _open_by_search(self, peer: str) -> bool:
        """用微信搜索（Cmd+F）打开会话；名称逐级截短，兼容 OCR 误差。"""
        press_command = getattr(self.platform, "press_command_key", None)
        if not callable(press_command):
            return False
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        queries = []
        for query in (peer, peer[:6], peer[:4], peer[:2]):
            if len(query) >= 2 and query not in queries:
                queries.append(query)
        for query in queries:
            try:
                press_command(3)  # Cmd+F
                time.sleep(0.35)
                if not self.platform.set_clipboard(query):
                    continue
                self.platform.press_ctrl_v()
                time.sleep(0.55)
                self.platform.press_enter()
                time.sleep(0.85)
                image = self.platform.capture()
                if image is None:
                    continue
                regions = self.layout.regions(image.width, image.height)
                if self._same_peer(self.sender.read_header(image, regions), peer):
                    return True
            except Exception:
                continue
        # 清掉搜索框，避免影响后续操作
        try:
            if callable(getattr(self.platform, "press_escape", None)):
                self.platform.press_escape()
        except Exception:
            pass
        return False

    def _open_by_position(self, peer: str) -> bool:
        position = self._peer_positions.get(peer)
        if not position:
            return False
        page = int(position.get("page", 0))
        y = float(position.get("y", 0))
        self._scroll_session_list(8, 14)   # 回到顶部
        for _ in range(page):
            self._scroll_session_list(-6, 1)
        image = self.platform.capture()
        if image is None:
            return False
        regions = self.layout.regions(image.width, image.height)
        left, _top, right, _bottom = regions.session_list
        x = left + (right - left) * 0.55
        for offset in (0, 12, -12, 24, -24):
            if hasattr(self.platform, "ensure_visible"):
                try:
                    self.platform.ensure_visible()
                except Exception:
                    pass
            self.platform.click(x, max(0, y + offset))
            time.sleep(0.8)
            current = self.platform.capture()
            if current is None:
                continue
            check = self.layout.regions(current.width, current.height)
            if self._same_peer(self.sender.read_header(current, check), peer):
                return True
        return False

    def _open_by_visible(self, peer: str) -> bool:
        """当前可见列表里直接找行点击（最稳，不需要搜索）。"""
        image = self.platform.capture()
        if image is None:
            return False
        regions = self.layout.regions(image.width, image.height)
        if self._same_peer(self.sender.read_header(image, regions), peer):
            return True
        try:
            lines = self.ocr.recognize(image.crop(regions.session_list))
        except Exception:
            return False
        target = None
        for line in lines:
            if self._same_peer(line.text, peer):
                target = line
                break
        if target is None:
            return False
        left, _top, right, _bottom = regions.session_list
        x = left + (right - left) * 0.55
        y = (target.bbox[1] + target.bbox[3]) / 2.0
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        self.platform.click(x, y)
        time.sleep(0.8)
        current = self.platform.capture()
        if current is None:
            return False
        check = self.layout.regions(current.width, current.height)
        return self._same_peer(self.sender.read_header(current, check), peer)

    def open_conversation(self, peer: str) -> bool:
        """打开任意会话：可见列表 → 滚动查找 → 搜索兜底。"""
        if self._open_by_visible(peer):
            return True
        if self._scroll_find_and_click(peer):
            return True
        if self._open_by_search(peer):
            return True
        image = self.platform.capture()
        if image is None:
            return False
        regions = self.layout.regions(image.width, image.height)
        if self._same_peer(self.sender.read_header(image, regions), peer):
            return True
        try:
            lines = self.ocr.recognize(image.crop(regions.session_list))
        except Exception:
            return False
        wanted = normalize_name(peer)
        best = None
        for line in lines:
            text = normalize_name(line.text)
            if text and (text == wanted or wanted in text or text in wanted):
                best = line
                break
        if best is None:
            # 列表里没有：滚动会话列表逐页找
            if self._scroll_find_and_click(peer):
                return True
            # 再尝试搜索框
            if self._search_and_open(image, regions, peer):
                return True
            return False
        y = (best.bbox[1] + best.bbox[3]) / 2.0
        x = regions.session_list[0] + (regions.session_list[2] - regions.session_list[0]) * 0.55
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        self.platform.click(x, y)
        time.sleep(0.8)
        current = self.platform.capture()
        if current is None:
            return False
        regions = self.layout.regions(current.width, current.height)
        return self._same_peer(self.sender.read_header(current, regions), peer)

    def _scroll_find_and_click(self, peer: str) -> bool:
        """滚动微信会话列表，找到目标会话并点开。"""
        wanted = normalize_name(peer)
        self._scroll_session_list(8, 10)  # 滚到顶部
        for _page in range(10):
            image = self.platform.capture()
            if image is None:
                return False
            regions = self.layout.regions(image.width, image.height)
            try:
                lines = self.ocr.recognize(image.crop(regions.session_list))
            except Exception:
                return False
            target = None
            for line in lines:
                text = normalize_name(line.text)
                if text and (text == wanted or wanted in text or text in wanted):
                    target = line
                    break
            if target is not None:
                y = (target.bbox[1] + target.bbox[3]) / 2.0
                x = regions.session_list[0] + (regions.session_list[2] - regions.session_list[0]) * 0.55
                if hasattr(self.platform, "ensure_visible"):
                    try:
                        self.platform.ensure_visible()
                    except Exception:
                        pass
                self.platform.click(x, y)
                time.sleep(0.9)
                current = self.platform.capture()
                if current is None:
                    return False
                check = self.layout.regions(current.width, current.height)
                if self._same_peer(self.sender.read_header(current, check), peer):
                    return True
            self._scroll_session_list(-4, 1)
        return False

    def _search_and_open(self, image, regions: Regions, peer: str) -> bool:
        """用微信搜索框搜索联系人并点开。"""
        left, _top, right, _bottom = regions.session_list
        width = max(1, right - left)
        search_x = left + width * 0.35
        search_y = 58
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        self.platform.click(search_x, search_y)
        time.sleep(0.3)
        if not self.platform.set_clipboard(peer):
            return False
        self.platform.press_ctrl_v()
        time.sleep(1.0)
        latest = self.platform.capture()
        if latest is None:
            return False
        try:
            lines = self.ocr.recognize(latest.crop((left, 0, right, latest.height)))
        except Exception:
            return False
        wanted = normalize_name(peer)
        best = None
        for line in lines:
            text = normalize_name(line.text)
            if text and (text == wanted or wanted in text):
                best = line
                break
        if best is None:
            return False
        y = (best.bbox[1] + best.bbox[3]) / 2.0 + 10
        self.platform.click(left + width * 0.55, y)
        time.sleep(0.9)
        current = self.platform.capture()
        if current is None:
            return False
        check_regions = self.layout.regions(current.width, current.height)
        return normalize_name(self.sender.read_header(current, check_regions)) == wanted

    # ---------- 手动：处理当前打开的会话（不点不点，只读+生成草稿） ----------
    def process_current(self) -> Optional[Dict[str, Any]]:
        image = self.platform.capture()
        if image is None:
            self.emit("error", message="抓图失败")
            return None
        regions = self.layout.regions(image.width, image.height)
        header = self.sender.read_header(image, regions)
        peer = header or ""
        if not peer:
            self.emit("error", message="读不到会话标题（OCR/权限问题）")
            return None
        items = self._read_items(image, regions)
        if not items:
            self.emit("skip", peer=peer, reason="当前会话没有可读消息")
            return None
        if is_group_name(peer):
            allowed, reason = group_allows(
                peer, items, self.runtime.allow_group_chat,
                self.runtime.group_whitelist, self.runtime.group_reply_mode,
                self.runtime.group_keywords,
            )
            if not allowed:
                self.emit("skip", peer=peer, reason=reason)
                return None
        else:
            allowed, _matched = self.whitelist.matches(peer)
            if not allowed:
                self.emit("skip", peer=peer, reason="不在白名单（当前会话）")
                return None
        incoming = [item.prompt_text() for item in items]
        joined = " / ".join(incoming)
        signature_now = normalize_name(joined)
        if self._replied.get(peer) == signature_now:
            self.emit("skip", peer=peer, reason="当前会话消息已处理过")
            return None
        history = self.memory.history_lines(peer, self.config.memory.recent_limit)
        profile = self.memory.profile(peer)
        for item in items:
            self.memory.add(peer, "in", item.kind, item.prompt_text())
        self.emit("incoming", peer=peer, messages=incoming,
                  kinds=[item.kind for item in items])
        signature = normalize_name(joined)
        decision = self.guardrails.assess_incoming(joined)
        if decision.action == "escalate":
            self.emit("escalate", peer=peer, reason=decision.reason)
            return None
        tool_results = self.tools.prefetch(joined)
        for result in tool_results:
            self.emit("tool", name=result.name, summary=result.summary, ok=result.ok)
        reply, model, latency = self._generate(incoming, joined, history, profile,
                                               tool_results, is_group_name(peer), peer=peer)
        if reply is None:
            return None
        self._pending[peer] = {
            "peer": peer, "reply": reply, "incoming": joined, "image": image,
            "regions": regions, "model": model, "latency_ms": latency,
            "incoming_items": incoming, "history": history, "profile": profile,
            "tool_results": [result.as_dict() for result in tool_results],
            "is_group": is_group_name(peer),
        }
        if peer in self._pending_order:
            self._pending_order.remove(peer)
        self._pending_order.append(peer)
        self._touch(peer, status="草稿待发", unread=0, text=reply)
        self.audit.record(peer=peer, action="draft", status="manual_draft",
                          incoming=joined, reply=reply, model=model,
                          latency_ms=latency, image=image)
        self.emit("draft", peer=peer, reply=reply, model=model,
                  latency_ms=latency, dry_run=self.dry_run,
                  tools=[result.name for result in tool_results])
        if self.dry_run:
            self.memory.add(peer, "out", "text", reply)
        return {"peer": peer, "reply": reply, "dry_run": self.dry_run,
                "incoming": incoming}

    # ---------- 人工介入 ----------
    def _resolve_pending(self, peer: Optional[str]) -> Optional[Dict[str, Any]]:
        if peer:
            return self._pending.get(peer)
        return self.pending

    def approve_pending(self, text_override: Optional[str] = None,
                        peer: Optional[str] = None) -> Optional[Dict[str, Any]]:
        pending = self._resolve_pending(peer)
        if not pending:
            return None
        peer = pending["peer"]
        reply = (text_override or "").strip() or pending["reply"]
        checked = self.guardrails.check_reply(reply)
        if checked.action != "reply":
            return {"ok": False, "status": "blocked", "detail": checked.reason,
                    "peer": peer, "reply": reply}
        pending["reply"] = checked.text
        # 如果用户已经切到别的会话，先切回目标会话再发，否则会被 header 校验中止
        current = self.platform.capture()
        if current is not None:
            regions = self.layout.regions(current.width, current.height)
            if normalize_name(self.sender.read_header(current, regions)) != normalize_name(peer):
                if not self.open_conversation(peer):
                    return {"ok": False, "status": "aborted",
                            "detail": "无法切回目标会话", "peer": peer,
                            "reply": pending["reply"]}
                pending["image"] = self.platform.capture()
                pending["regions"] = self.layout.regions(
                    pending["image"].width, pending["image"].height
                ) if pending["image"] is not None else pending["regions"]
        if pending["image"] is None:
            pending["image"] = self.platform.capture()
        outcome = self.sender.send(peer, pending["reply"], pending["image"], pending["regions"])
        signature = normalize_name(pending.get("incoming", ""))
        ok = self._finish_send(peer, pending["reply"], outcome, pending.get("incoming", ""),
                               pending.get("model", ""), pending.get("latency_ms", 0), signature)
        return {"ok": ok, "status": outcome.outcome.status, "detail": outcome.outcome.detail,
                "peer": peer, "reply": pending["reply"]}

    def discard_pending(self, peer: Optional[str] = None) -> None:
        target = self._resolve_pending(peer)
        if target:
            self._pending.pop(target["peer"], None)
            if target["peer"] in self._pending_order:
                self._pending_order.remove(target["peer"])
        self.emit("status", message="已丢弃当前草稿")

    def regenerate_pending(self, peer: Optional[str] = None) -> Optional[Dict[str, Any]]:
        pending = self._resolve_pending(peer)
        if not pending:
            return None
        reply, model, latency = self._generate(
            pending.get("incoming_items") or [pending.get("incoming", "")],
            pending.get("incoming", ""),
            pending.get("history") or [],
            pending.get("profile") or {},
            pending.get("tool_results") or [],
            pending.get("is_group", False),
            peer=pending.get("peer", ""),
        )
        if reply is None:
            return None
        pending.update({"reply": reply, "model": model, "latency_ms": latency})
        self.emit("draft", peer=pending["peer"], reply=reply, model=model,
                  latency_ms=latency, dry_run=self.dry_run, regenerated=True)
        return {"reply": reply, "peer": pending["peer"]}

    def send_manual(self, peer: str, text: str) -> Optional[Dict[str, Any]]:
        image = self.platform.capture()
        if image is None:
            return {"ok": False, "status": "capture_failed"}
        regions = self.layout.regions(image.width, image.height)
        checked = self.guardrails.check_reply(text)
        if checked.action != "reply":
            return {"ok": False, "status": "blocked", "detail": checked.reason}
        outcome = self.sender.send(peer, checked.text, image, regions)
        if outcome.outcome.status == "verified":
            self.memory.add(peer, "out", "text", checked.text)
            self._touch(peer, status="已发送", text=checked.text)
        return {"ok": outcome.outcome.status == "verified",
                "status": outcome.outcome.status, "detail": outcome.outcome.detail}

    def close(self) -> None:
        self.audit.close()
        self.memory.close()
