import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { DirectChatMessage } from "./types";

const ESTIMATED_MESSAGE_HEIGHT_PX = 120;
const ESTIMATED_PENDING_HEIGHT_PX = 72;
const OVERSCAN_COUNT = 6;
const STICK_TO_BOTTOM_THRESHOLD_PX = 96;

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
    getItemKey: (index) => rows[index]?.key ?? index,
  });

  const virtualItems = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();

  const updateStickToBottom = useCallback(() => {
    const element = parentRef.current;
    if (!element) return;
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    stickToBottomRef.current = distanceFromBottom <= STICK_TO_BOTTOM_THRESHOLD_PX;
  }, []);

  useEffect(() => {
    const element = parentRef.current;
    if (!element) return;

    updateStickToBottom();
    element.addEventListener("scroll", updateStickToBottom, { passive: true });
    return () => {
      element.removeEventListener("scroll", updateStickToBottom);
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
    requestAnimationFrame(() => {
      virtualizer.scrollToIndex(lastIndex, { align: "end", behavior: "auto" });
      stickToBottomRef.current = true;
    });
  }, [rows.length, virtualizer]);

  // Keep the viewport pinned after dynamic Markdown measurement when the user is following the latest messages.
  useLayoutEffect(() => {
    if (rows.length === 0 || !stickToBottomRef.current) return;
    const element = parentRef.current;
    if (!element) return;

    requestAnimationFrame(() => {
      if (!stickToBottomRef.current || !parentRef.current) return;
      parentRef.current.scrollTop = parentRef.current.scrollHeight;
    });
  }, [rows.length, totalSize]);

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
