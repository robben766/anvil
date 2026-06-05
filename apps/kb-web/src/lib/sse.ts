import type { QueryDone, Source } from "./types";

const BASE = process.env.NEXT_PUBLIC_KB_API_URL ?? "http://localhost:8400";

export interface SseCallbacks {
  onSources(s: Source[]): void;
  onDelta(text: string): void;
  onDone(d: QueryDone): void;
  onError(e: Error): void;
}

/**
 * Streams a KB query via POST SSE.
 * Returns an abort function; calling it cancels the in-flight request.
 *
 * SSE frame format (each frame terminated by "\n\n"):
 *   event: <name>\ndata: <single-line-json>\n\n
 *
 * Events: sources | delta | done
 */
export function streamQuery(
  question: string,
  k: number,
  cb: SseCallbacks,
): () => void {
  const controller = new AbortController();

  (async () => {
    let resp: Response;
    try {
      resp = await fetch(`${BASE}/v1/kb/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k, stream: true }),
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
          dispatchFrame(frame.trim(), cb);
        }
      }

      // Flush remaining decoder bytes (stream: false signals end)
      buffer += decoder.decode(undefined, { stream: false });
      // Process any final frame that wasn't followed by "\n\n"
      if (buffer.trim()) {
        dispatchFrame(buffer.trim(), cb);
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      cb.onError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      reader.releaseLock();
    }
  })();

  return () => controller.abort();
}

/** Parse a single SSE frame and dispatch to the appropriate callback. */
function dispatchFrame(frame: string, cb: SseCallbacks): void {
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

  if (!dataLine) return;

  let parsed: unknown;
  try {
    parsed = JSON.parse(dataLine);
  } catch {
    // Malformed JSON — skip
    return;
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
      break;
    // Unknown events are silently ignored
  }
}
