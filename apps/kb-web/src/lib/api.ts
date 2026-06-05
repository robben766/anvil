const BASE = process.env.NEXT_PUBLIC_KB_API_URL ?? "http://localhost:8400";

export interface DocumentSummary {
  id: string;
  title: string;
  source_name: string;
  chunk_count: number;
  created_at: string;
}

export interface DocumentDetail extends DocumentSummary {
  content: string;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore parse errors
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const res = await fetch(`${BASE}/documents`);
  return handleResponse<DocumentSummary[]>(res);
}

export async function uploadDocument(
  file: File,
): Promise<{ id: string; title: string; source_name: string; chunk_count: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents`, { method: "POST", body: form });
  return handleResponse<{
    id: string;
    title: string;
    source_name: string;
    chunk_count: number;
  }>(res);
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore parse errors
    }
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
}

export async function getDocument(id: string): Promise<DocumentDetail> {
  const res = await fetch(`${BASE}/documents/${encodeURIComponent(id)}`);
  return handleResponse<DocumentDetail>(res);
}
