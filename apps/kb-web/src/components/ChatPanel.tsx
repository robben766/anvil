"use client";

import { useRef, useState } from "react";
import { streamQuery } from "@/lib/sse";
import type { Citation, DebugFrame, QueryDone, Source } from "@/lib/types";
import DebugPanel from "./DebugPanel";

/** A single answered turn kept in history. */
interface Turn {
  id: number;
  question: string;
  /** Accumulated streamed answer text. */
  answer: string;
  sources: Source[];
  citations: Citation[];
  /** Still streaming? */
  streaming: boolean;
  error: string | null;
  /** Sources list is expanded? */
  sourcesOpen: boolean;
  /** Debug frame received from SSE `debug` event (only when debug mode on). */
  debugFrame: DebugFrame | null;
  /** Debug panel is expanded? (defaults true when frame arrives) */
  debugOpen: boolean;
}

interface Props {
  /** Called when the user clicks a [n] citation superscript. */
  onCite(citation: Citation): void;
}

export default function ChatPanel({ onCite }: Props) {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streaming, setStreaming] = useState(false);
  // Debug mode persists across turns (user-controlled checkbox)
  const [debugMode, setDebugMode] = useState(false);
  // Rerank mode persists across turns (user-controlled checkbox, default off)
  const [rerankMode, setRerankMode] = useState(false);
  const abortRef = useRef<(() => void) | null>(null);
  const nextId = useRef(0);

  function startQuery() {
    const q = question.trim();
    if (!q || streaming) return;
    setQuestion("");

    const id = nextId.current++;
    const newTurn: Turn = {
      id,
      question: q,
      answer: "",
      sources: [],
      citations: [],
      streaming: true,
      error: null,
      sourcesOpen: false,
      debugFrame: null,
      debugOpen: true,
    };
    setTurns((prev) => [...prev, newTurn]);
    setStreaming(true);

    function patchTurn(patch: Partial<Turn>) {
      setTurns((prev) =>
        prev.map((t) => (t.id === id ? { ...t, ...patch } : t)),
      );
    }

    const abort = streamQuery(
      q,
      5,
      {
        onSources(s) {
          patchTurn({ sources: s });
        },
        onDelta(text) {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === id ? { ...t, answer: t.answer + text } : t,
            ),
          );
        },
        onDone(d: QueryDone) {
          // done.text 是后端保证的完整答案(B2 测试断言 delta 拼接 == done.text),覆盖累计值以消除任何流式拼接误差
          patchTurn({
            answer: d.text,
            citations: d.citations,
            streaming: false,
          });
          setStreaming(false);
          abortRef.current = null;
        },
        onError(e) {
          patchTurn({ error: e.message, streaming: false });
          setStreaming(false);
          abortRef.current = null;
        },
        onDebug(frame) {
          patchTurn({ debugFrame: frame, debugOpen: true });
        },
        onStreamEnd(receivedDone) {
          // 断流兜底: 流自然结束但未收到 done 事件 → 标记为意外中断
          // 仅在该轮无既有 error 时设置,避免覆盖 error 帧带来的真实错误信息
          if (!receivedDone) {
            setTurns((prev) =>
              prev.map((t) =>
                t.id === id && t.streaming && !t.error
                  ? { ...t, error: "流意外中断", streaming: false }
                  : t,
              ),
            );
            setStreaming(false);
            abortRef.current = null;
          }
        },
      },
      debugMode,
      rerankMode,
    );

    abortRef.current = abort;
  }

  function handleStop() {
    abortRef.current?.();
    abortRef.current = null;
    // Mark the currently-streaming turn as no longer streaming
    setTurns((prev) =>
      prev.map((t) => (t.streaming ? { ...t, streaming: false } : t)),
    );
    setStreaming(false);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      startQuery();
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* History of turns */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
        {turns.length === 0 && (
          <div className="flex h-full items-center justify-center text-sm text-slate-400">
            <p>输入问题开始问答</p>
          </div>
        )}

        {turns.map((turn) => (
          <TurnCard
            key={turn.id}
            turn={turn}
            onCite={onCite}
            onToggleSources={() =>
              setTurns((prev) =>
                prev.map((t) =>
                  t.id === turn.id
                    ? { ...t, sourcesOpen: !t.sourcesOpen }
                    : t,
                ),
              )
            }
            onToggleDebug={() =>
              setTurns((prev) =>
                prev.map((t) =>
                  t.id === turn.id
                    ? { ...t, debugOpen: !t.debugOpen }
                    : t,
                ),
              )
            }
          />
        ))}
      </div>

      {/* Input area */}
      <div className="shrink-0 border-t border-slate-200 px-4 py-3">
        <div className="flex gap-2 items-end">
          <textarea
            className="flex-1 resize-none rounded border border-slate-300 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-blue-400 focus:outline-none disabled:opacity-50"
            rows={2}
            placeholder="输入问题，Enter 发送，Shift+Enter 换行"
            value={question}
            disabled={streaming}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <div className="flex flex-col gap-2 items-end">
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={rerankMode}
                  onChange={(e) => setRerankMode(e.target.checked)}
                  className="rounded"
                />
                重排
              </label>
              <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={debugMode}
                  onChange={(e) => setDebugMode(e.target.checked)}
                  className="rounded"
                />
                调试
              </label>
            </div>
            {streaming ? (
              <button
                onClick={handleStop}
                className="shrink-0 rounded bg-red-100 px-3 py-2 text-sm font-medium text-red-700 hover:bg-red-200"
              >
                停止
              </button>
            ) : (
              <button
                onClick={startQuery}
                disabled={!question.trim()}
                className="shrink-0 rounded bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
              >
                发送
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---- Sub-components ----

interface TurnCardProps {
  turn: Turn;
  onCite(c: Citation): void;
  onToggleSources(): void;
  onToggleDebug(): void;
}

function TurnCard({ turn, onCite, onToggleSources, onToggleDebug }: TurnCardProps) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white shadow-sm">
      {/* Question header */}
      <div className="border-b border-slate-100 px-4 py-2.5 text-sm font-medium text-slate-700 bg-slate-50 rounded-t-lg">
        {turn.question}
      </div>

      <div className="px-4 py-3 space-y-3">
        {/* Debug panel — appears above sources, collapsible, default expanded */}
        {turn.debugFrame && (
          <div>
            <button
              onClick={onToggleDebug}
              className="flex items-center gap-1 text-xs font-medium text-indigo-600 hover:text-indigo-900"
            >
              <span
                className={`transition-transform ${turn.debugOpen ? "rotate-90" : ""}`}
              >
                ▶
              </span>
              检索调试（dense {turn.debugFrame.dense.length} / sparse{" "}
              {turn.debugFrame.sparse.length} / fused {turn.debugFrame.fused.length}
              {turn.debugFrame.reranked !== null
                ? ` / reranked ${turn.debugFrame.reranked.length}`
                : ""}）
            </button>
            {turn.debugOpen && (
              <div className="mt-2">
                <DebugPanel frame={turn.debugFrame} />
              </div>
            )}
          </div>
        )}

        {/* Sources collapsible */}
        {turn.sources.length > 0 && (
          <div>
            <button
              onClick={onToggleSources}
              className="flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-slate-800"
            >
              <span
                className={`transition-transform ${turn.sourcesOpen ? "rotate-90" : ""}`}
              >
                ▶
              </span>
              检索来源（{turn.sources.length} 条）
            </button>

            {turn.sourcesOpen && (
              <ul className="mt-1.5 space-y-1 pl-4">
                {turn.sources.map((src) => (
                  <li key={src.chunk_id} className="text-xs text-slate-500">
                    <span className="font-medium text-slate-700">
                      [{src.n}]
                    </span>{" "}
                    {src.header_path} · {src.score.toFixed(3)}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Error */}
        {turn.error && (
          <div className="rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
            {turn.error}
          </div>
        )}

        {/* Answer area */}
        {(turn.answer || turn.streaming) && (
          <div className="font-mono text-sm whitespace-pre-wrap leading-relaxed text-slate-800">
            {turn.streaming ? (
              // Still streaming: show raw text + blinking cursor
              <>
                {turn.answer}
                <span className="animate-pulse">▌</span>
              </>
            ) : (
              // Done: render [n] as clickable superscripts
              <AnnotatedAnswer
                text={turn.answer}
                citations={turn.citations}
                onCite={onCite}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface AnnotatedAnswerProps {
  text: string;
  citations: Citation[];
  onCite(c: Citation): void;
}

/**
 * Splits the answer text by `[n]` citation markers and renders them as
 * clickable superscript buttons. Only markers present in `citations` are
 * made interactive. No dangerouslySetInnerHTML is used.
 */
function AnnotatedAnswer({ text, citations, onCite }: AnnotatedAnswerProps) {
  const citationMap = new Map<number, Citation>(
    citations.map((c) => [c.n, c]),
  );

  // Split by [n] patterns; capture the n value
  const CITATION_RE = /\[(\d+)\]/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = CITATION_RE.exec(text)) !== null) {
    const n = Number(match[1]);
    const citation = citationMap.get(n);

    // Append the plain text before this marker
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    if (citation) {
      // Render as a clickable superscript
      parts.push(
        <button
          key={`${n}-${match.index}`}
          onClick={() => onCite(citation)}
          className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded bg-blue-100 px-0.5 text-[10px] font-semibold text-blue-700 hover:bg-blue-200 align-super"
          aria-label={`引用 ${n}`}
          title={citation.header_path}
        >
          {n}
        </button>,
      );
    } else {
      // Not in citations — keep as plain text
      parts.push(match[0]);
    }

    lastIndex = match.index + match[0].length;
  }

  // Append remaining text after the last marker
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return <>{parts}</>;
}
