"use client";

import { useEffect, useRef, useState } from "react";
import {
  deleteDocument,
  listDocuments,
  uploadDocument,
  type DocumentSummary,
} from "@/lib/api";

export default function DocumentPanel() {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [uploading, setUploading] = useState(false);
  const [lastChunkCount, setLastChunkCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  // Fetch document list whenever refreshKey changes
  useEffect(() => {
    let cancelled = false;
    listDocuments()
      .then((list) => {
        if (!cancelled) setDocs(list);
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  function refresh() {
    setRefreshKey((k) => k + 1);
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    setLastChunkCount(null);
    try {
      const result = await uploadDocument(file);
      setLastChunkCount(result.chunk_count);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function handleDelete(id: string, title: string) {
    if (!confirm(`确认删除文档「${title}」？`)) return;
    setError(null);
    try {
      await deleteDocument(id);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function formatDate(iso: string) {
    try {
      return new Date(iso).toLocaleDateString("zh-CN", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
      });
    } catch {
      return iso;
    }
  }

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Error bar */}
      {error && (
        <div className="flex items-center justify-between gap-2 rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
          <span>{error}</span>
          <button
            onClick={() => setError(null)}
            className="shrink-0 font-bold hover:text-red-900"
            aria-label="关闭错误提示"
          >
            ✕
          </button>
        </div>
      )}

      {/* Upload section */}
      <div className="flex flex-col gap-2 rounded border border-slate-200 bg-slate-50 p-3">
        <label className="text-sm font-medium text-slate-700">上传文档</label>
        <input
          ref={fileRef}
          type="file"
          accept=".md,.txt,.pdf"
          disabled={uploading}
          onChange={handleUpload}
          className="text-sm text-slate-600 file:mr-2 file:cursor-pointer file:rounded file:border file:border-slate-300 file:bg-white file:px-2 file:py-1 file:text-xs file:text-slate-700 disabled:opacity-50"
        />
        <p className="text-xs text-slate-400">支持 .md、.txt、.pdf 格式，最大 2 MB</p>
        {uploading && (
          <p className="text-xs text-slate-500">上传中…</p>
        )}
        {lastChunkCount !== null && !uploading && (
          <p className="text-xs text-green-600">
            上传成功，共切分 {lastChunkCount} 个 chunk
          </p>
        )}
      </div>

      {/* Document list */}
      <div className="flex flex-col gap-2 overflow-y-auto">
        {docs.length === 0 ? (
          <p className="text-sm text-slate-400">暂无文档</p>
        ) : (
          docs.map((doc) => (
            <div
              key={doc.id}
              className="flex items-start justify-between gap-2 rounded border border-slate-200 bg-white px-3 py-2"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-800">
                  {doc.title}
                </p>
                <p className="mt-0.5 text-xs text-slate-500">
                  {doc.chunk_count} chunks · {formatDate(doc.created_at)}
                </p>
              </div>
              <button
                onClick={() => handleDelete(doc.id, doc.title)}
                className="shrink-0 rounded px-2 py-1 text-xs text-red-600 hover:bg-red-50 hover:text-red-800"
              >
                删除
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
