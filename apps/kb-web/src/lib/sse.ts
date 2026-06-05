import type { DebugFrame, QueryDone, Source } from "./types";
import { BASE } from "./api";

export interface SseCallbacks {
  onSources(s: Source[]): void;
  onDelta(text: string): void;
  onDone(d: QueryDone): void;
  onError(e: Error): void;
  /** Called when the SSE `debug` event is received (only when debug:true). */
  onDebug?(d: DebugFrame): void;
  /**
   * Called when the stream ends (reader done) whether or not a `done` event
   * was received. Use this for "stream-break guard": if the stream ended
   * without a `done` event the caller can mark the turn as broken.
   * @param receivedDone  true if a `done` event was dispatched before EOF
   */
  onStreamEnd?(receivedDone: boolean): void;
}

/**
 * Streams a KB query via POST SSE.
 * Returns an abort function; calling it cancels the in-flight request.
 *
 * SSE frame format (each frame terminated by "\n\n"):
 *   event: <name>\ndata: <single-line-json>\n\n
 *
 * Events: sources | delta | done | debug | error
 */
export function streamQuery(
  question: string,
  k: number,
  cb: SseCallbacks,
  debug?: boolean,
  rerank?: boolean,
): () => void {
  const controller = new AbortController();

  (async () => {
    let resp: Response;
    try {
      resp = await fetch(`${BASE}/v1/kb/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          k,
          stream: true,
          ...(debug ? { debug: true } : {}),
          ...(rerank ? { rerank: true } : {}),
        }),
        signal: controller.signal,
      });
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      cb.onError(err instanceof Error ? err : new Error(String(err)));
      return;
    }

    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const body = await resp.json();
        if (body?.detail) detail = String(body.detail);
      } catch {
        // ignore parse errors
      }
      cb.onError(new Error(`HTTP ${resp.status}: ${detail}`));
      return;
    }

    const reader = resp.body?.getReader();
    if (!reader) {
      cb.onError(new Error("Response body is not readable"));
      return;
    }

    const decoder = new TextDecoder();
    // Buffer for incomplete frames across chunks
    let buffer = "";
    // Track whether a `done` event was dispatched (for stream-break guard)
    let receivedDone = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Split on double-newline (SSE frame boundary)
        const frames = buffer.split("\n\n");
        // Last element is the incomplete (possibly empty) tail — keep in buffer
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          if (!frame.trim()) continue;
          if (dispatchFrame(frame.trim(), cb)) receivedDone = true;
        }
      }

      // Flush remaining decoder bytes (stream: false signals end)
      buffer += decoder.decode(undefined, { stream: false });
      // Process any final frame that wasn't followed by "\n\n"
      if (buffer.trim()) {
        if (dispatchFrame(buffer.trim(), cb)) receivedDone = true;
      }

      // Stream ended naturally — notify caller so it can handle断流兜底
      cb.onStreamEnd?.(receivedDone);
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        reader.cancel().catch(() => {});
        return;
      }
      cb.onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      reader.releaseLock();
    }
  })();

  return () => controller.abort();
}

/**
 * Parse a single SSE frame and dispatch to the appropriate callback.
 * Returns true if the `done` event was dispatched (signals clean stream end).
 */
function dispatchFrame(frame: string, cb: SseCallbacks): boolean {
  const lines = frame.split("\n");
  let eventName = "";
  let dataLine = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLine = line.slice("data:".length).trim();
    }
  }

  if (!dataLine) return false;

  let parsed: unknown;
  try {
    parsed = JSON.parse(dataLine);
  } catch {
    // Malformed JSON — skip
    return false;
  }

  switch (eventName) {
    case "sources":
      cb.onSources(parsed as Source[]);
      break;
    case "delta":
      cb.onDelta((parsed as { text: string }).text);
      break;
    case "done":
      cb.onDone(parsed as QueryDone);
      return true;
    case "debug":
      cb.onDebug?.(parsed as DebugFrame);
      break;
    case "error":
      cb.onError(new Error((parsed as { detail: string }).detail));
      break;
    // Unknown events are silently ignored
  }
  return false;
}
