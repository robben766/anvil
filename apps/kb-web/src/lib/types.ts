/** Source returned in the SSE `sources` event (one per retrieved chunk). */
export interface Source {
  /** 1-based index used for [n] references in the answer. */
  n: number;
  chunk_id: string;
  document_id: string;
  /** Verbatim excerpt from the chunk. */
  quote: string;
  /** Breadcrumb path of headings, e.g. "# Title > ## Section". */
  header_path: string;
  start_offset: number;
  end_offset: number;
  score: number;
}

/** Citation referenced in the SSE `done` event (subset of Source without score). */
export interface Citation {
  n: number;
  chunk_id: string;
  document_id: string;
  quote: string;
  header_path: string;
  start_offset: number;
  end_offset: number;
}

/** Payload of the SSE `done` event. */
export interface QueryDone {
  text: string;
  citations: Citation[];
}

/** A single item in dense/sparse/fused arrays of the SSE `debug` frame. */
export interface DebugItem {
  n: number;
  chunk_id: string;
  quote_head: string;
  score: number;
  rank: number;
  /** Only present in fused items: original ranks from each retriever. */
  contributions?: {
    dense: number | null;
    sparse: number | null;
  };
}

/** Payload of the SSE `debug` event (arrives before `sources`). */
export interface DebugFrame {
  dense: DebugItem[];
  sparse: DebugItem[];
  fused: DebugItem[];
  /** Cross-encoder reranked order; null when rerank was not requested. */
  reranked: DebugItem[] | null;
}
