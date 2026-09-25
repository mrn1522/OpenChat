import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { DirectChatMessage } from "./types";

const ESTIMATED_MESSAGE_HEIGHT_PX = 120;
const ESTIMATED_PENDING_HEIGHT_PX = 72;
const OVERSCAN_COUNT = 6;
const STICK_TO_BOTTOM_THRESHOLD_PX = 96;
const BOTTOM_SCROLL_GAP_PX = 40;

type TranscriptRow =
  | { kind: "message"; key: string; message: DirectChatMessage; index: number }
  | { kind: "pending"; key: string };

type DirectChatTranscriptProps = {
  messages: DirectChatMessage[];
  isRunning: boolean;
};

const DirectChatMessageBubble = memo(function DirectChatMessageBubble({
  message,
}: {
  message: DirectChatMessage;
}) {
  return (
    <article className={`direct-chat-bubble ${message.role}`}>
      <p className="direct-chat-role">{message.role === "user" ? "You" : "Assistant"}</p>
      {message.images && message.images.length > 0 && (
        <div className="direct-chat-image-row">
          {message.images.map((image, index) =>
            image.data_url ? (
              <img
                key={`${image.name}-${index}`}
                className="direct-chat-image-thumb"
                src={image.data_url}
                alt={image.name}
              />
            ) : (
              <span key={`${image.name}-${index}`} className="direct-chat-image-chip">
                {image.name}
              </span>
            )
          )}
        </div>
      )}
      <div className="markdown-body">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
      </div>
    </article>
  );
});

const DirectChatPendingBubble = memo(function DirectChatPendingBubble() {
  return (
    <article className="direct-chat-bubble assistant pending" aria-label="Assistant is generating a response">
      <p className="direct-chat-role">Assistant</p>
      <div className="direct-chat-loading">
        <span className="loader" />
        <span>Thinking...</span>
      </div>
    </article>
  );
});

function DirectChatTranscript({ messages, isRunning }: DirectChatTranscriptProps) {
  const parentRef = useRef<HTMLDivElement | null>(null);
  const stickToBottomRef = useRef(true);
  const previousCountRef = useRef(0);
  const programmaticScrollRef = useRef(false);

  const rows = useMemo<TranscriptRow[]>(() => {
    const next: TranscriptRow[] = messages.map((message, index) => ({
      kind: "message",
      key: `message-${index}-${message.role}`,
      message,
      index,
    }));

    if (isRunning) {
      next.push({ kind: "pending", key: "pending-assistant" });
    }

    return next;
  }, [isRunning, messages]);

  const estimateSize = useCallback(
    (index: number) => {
      const row = rows[index];
      if (!row) return ESTIMATED_MESSAGE_HEIGHT_PX;
      if (row.kind === "pending") return ESTIMATED_PENDING_HEIGHT_PX;
      const contentLength = row.message.content.length;
      const estimatedBody = Math.min(720, Math.max(64, Math.ceil(contentLength / 90) * 22));
      return estimatedBody + 48;
    },
    [rows]
  );

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize,
    overscan: OVERSCAN_COUNT,
    paddingEnd: BOTTOM_SCROLL_GAP_PX,
    scrollPaddingEnd: BOTTOM_SCROLL_GAP_PX,
    getItemKey: (index) => rows[index]?.key ?? index,
  });

  const virtualItems = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();

  const smoothScrollBehavior = useCallback((): ScrollBehavior => {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  }, []);

  const updateStickToBottom = useCallback(() => {
    const element = parentRef.current;
    if (!element) return;
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    if (distanceFromBottom <= STICK_TO_BOTTOM_THRESHOLD_PX) {
      stickToBottomRef.current = true;
      programmaticScrollRef.current = false;
      return;
    }
    // Positions crossed mid-smooth-scroll are animation frames, not the
    // user scrolling away; wheel/pointer input cancels the flag below.
    if (programmaticScrollRef.current) return;
    stickToBottomRef.current = false;
  }, []);

  useEffect(() => {
    const element = parentRef.current;
    if (!element) return;

    const cancelProgrammaticScroll = () => {
      programmaticScrollRef.current = false;
    };
    const cancelProgrammaticScrollOnKeydown = (event: KeyboardEvent) => {
      if (["ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", " "].includes(event.key)) {
        cancelProgrammaticScroll();
      }
    };

    updateStickToBottom();
    element.addEventListener("scroll", updateStickToBottom, { passive: true });
    element.addEventListener("wheel", cancelProgrammaticScroll, { passive: true });
    element.addEventListener("pointerdown", cancelProgrammaticScroll);
    element.addEventListener("keydown", cancelProgrammaticScrollOnKeydown);
    return () => {
      element.removeEventListener("scroll", updateStickToBottom);
      element.removeEventListener("wheel", cancelProgrammaticScroll);
      element.removeEventListener("pointerdown", cancelProgrammaticScroll);
      element.removeEventListener("keydown", cancelProgrammaticScrollOnKeydown);
    };
  }, [updateStickToBottom]);

  useLayoutEffect(() => {
    if (rows.length === 0) {
      previousCountRef.current = 0;
      stickToBottomRef.current = true;
      return;
    }

    const previousCount = previousCountRef.current;
    const didGrow = rows.length > previousCount;
    previousCountRef.current = rows.length;

    if (!didGrow && previousCount !== 0) return;
    if (!stickToBottomRef.current && previousCount !== 0) return;

    const lastIndex = rows.length - 1;
    // First render jumps; later growth (new turns, pending bubble) eases down
    // so the conversation advances naturally.
    const behavior = previousCount === 0 ? "auto" : smoothScrollBehavior();
    requestAnimationFrame(() => {
      programmaticScrollRef.current = true;
      virtualizer.scrollToIndex(lastIndex, { align: "end", behavior });
      stickToBottomRef.current = true;
    });
  }, [rows.length, virtualizer, smoothScrollBehavior]);

  // Keep the viewport pinned after dynamic Markdown measurement when the user is following the latest messages.
  useLayoutEffect(() => {
    if (rows.length === 0 || !stickToBottomRef.current) return;
    const element = parentRef.current;
    if (!element) return;

    requestAnimationFrame(() => {
      const element = parentRef.current;
      if (!stickToBottomRef.current || !element) return;
      programmaticScrollRef.current = true;
      element.scrollTo({ top: element.scrollHeight, behavior: smoothScrollBehavior() });
    });
  }, [rows.length, totalSize, smoothScrollBehavior]);

  if (rows.length === 0) {
    return (
      <div className="direct-chat-transcript" role="log" aria-live="polite" aria-label="Direct chat messages">
        <div className="direct-chat-empty">Send a prompt to begin a direct conversation.</div>
      </div>
    );
  }

  return (
    <div
      ref={parentRef}
      className="direct-chat-transcript direct-chat-transcript-virtual"
      role="log"
      aria-live="polite"
      aria-label="Direct chat messages"
    >
      <div className="direct-chat-virtual-spacer" style={{ height: totalSize }}>
        {virtualItems.map((virtualRow) => {
          const row = rows[virtualRow.index];
          if (!row) return null;

          return (
            <div
              key={row.key}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              className="direct-chat-virtual-row"
              style={{
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              {row.kind === "message" ? (
                <DirectChatMessageBubble message={row.message} />
              ) : (
                <DirectChatPendingBubble />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default memo(DirectChatTranscript);
