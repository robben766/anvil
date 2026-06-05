"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { getDocument } from "@/lib/api";
import type { Citation } from "@/lib/types";

interface Props {
  citation: Citation | null;
  onClose(): void;
}

interface DocCache {
  /** The document_id this data belongs to. */
  documentId: string;
  loading: boolean;
  error: string | null;
  title: string;
  content: string;
}

export default function CitationPanel({ citation, onClose }: Props) {
  const [cache, setCache] = useState<DocCache | null>(null);
  const highlightRef = useRef<HTMLSpanElement>(null);
  // Tracks the document_id that is currently loaded or being fetched,
  // avoiding stale-closure reads of `cache` state inside the effect.
  const fetchedDocRef = useRef<string | null>(null);

  // Fetch document whenever citation's document_id changes
  useEffect(() => {
    if (!citation) return;

    const docId = citation.document_id;

    // Same doc already loaded or in flight — skip
    if (fetchedDocRef.current === docId) return;

    fetchedDocRef.current = docId;
    let cancelled = false;
    getDocument(docId)
      .then((doc) => {
        if (!cancelled) {
          setCache({
            documentId: docId,
            loading: false,
            error: null,
            title: doc.title,
            content: doc.content,
          });
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          // Clear ref so a subsequent click on the same doc can retry
          fetchedDocRef.current = null;
          setCache({
            documentId: docId,
            loading: false,
            error: err instanceof Error ? err.message : String(err),
            title: "",
            content: "",
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [citation]);

  // Scroll highlight into view after the DOM has been updated
  useLayoutEffect(() => {
    if (highlightRef.current) {
      highlightRef.current.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [cache?.content, citation]);

  if (!citation) return null;

  // Determine display state
  const loading = !cache || cache.documentId !== citation.document_id || cache.loading;
  const error = cache?.documentId === citation.document_id ? cache.error : null;
  const content = cache?.documentId === citation.document_id ? cache.content : "";
  const title = cache?.documentId === citation.document_id ? cache.title : "";

  return (
    // Fixed overlay panel on the right
    <div className="fixed right-0 top-0 z-50 flex h-full w-[28rem] flex-col border-l border-slate-200 bg-white shadow-xl">
      {/* Header */}
      <div className="flex shrink-0 items-start justify-between gap-3 border-b border-slate-200 px-4 py-3">
        <div className="min-w-0 flex-1">
          {title && (
            <p className="truncate text-sm font-semibold text-slate-800">
              {title}
            </p>
          )}
          <p className="mt-0.5 truncate text-xs text-slate-500">
            [{citation.n}] {citation.header_path}
          </p>
        </div>
        <button
          onClick={onClose}
          className="shrink-0 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          aria-label="关闭引用面板"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <line x1="3" y1="3" x2="13" y2="13" />
            <line x1="13" y1="3" x2="3" y2="13" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-4 py-3 text-sm leading-relaxed">
        {loading && (
          <p className="text-slate-400">加载中…</p>
        )}

        {error && (
          <p className="rounded border border-red-300 bg-red-50 px-3 py-2 text-red-700">
            {error}
          </p>
        )}

        {!loading && !error && content && (
          <p className="whitespace-pre-wrap break-words font-mono text-xs">
            {/* Before highlight */}
            <span className="text-slate-400">
              {content.slice(0, citation.start_offset)}
            </span>
            {/* Highlight segment — scroll target */}
            <span
              ref={highlightRef}
              className="rounded bg-yellow-200 px-0.5"
            >
              {content.slice(citation.start_offset, citation.end_offset)}
            </span>
            {/* After highlight */}
            <span className="text-slate-400">
              {content.slice(citation.end_offset)}
            </span>
          </p>
        )}
      </div>
    </div>
  );
}
