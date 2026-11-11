/**
 * The service's payload, typed, and the two calls that fetch it.
 *
 * These types are a transcription of `api.py`'s response, not an interpretation of
 * it. In particular `crossesBranches` is read, never computed: the branch a node
 * sits under is known on the server and nowhere else, and recomputing the project's
 * central claim from whatever happened to be serialised would get it subtly wrong
 * while still drawing a picture.
 */

export type NodeKind = "folder" | "document" | "chunk" | "entity";
export type EdgeKind = "contains" | "has_chunk" | "mentions" | "relates";
export type Legs = "both" | "search" | "traversal";
export type Mode = "hybrid" | "vector" | "fulltext";

export interface Ancestor {
  name: string;
  level: string;
}

/** Everything a node of any kind may carry. Kind-specific fields are optional
 * because the server sends one flat shape and switches on `kind`; so do we. */
export interface SubgraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  branches: string[];
  /** This node alone spans two branches. Only an entity ever can. */
  bridge: boolean;
  ancestry: Ancestor[];
  ancestry_lines: string[];
  // chunk
  source?: number;
  cited?: boolean;
  chunk_id?: string;
  document?: string;
  filename?: string;
  index?: number;
  page?: number | null;
  text?: string;
  score?: number;
  legs?: string[];
  ranks?: Record<string, number>;
  hop?: number | null;
  // folder
  depth?: number;
  level?: string;
  // document
  path?: string;
  // entity
  type?: string;
  scope?: string;
  normalized_name?: string;
  mentions?: number;
  matched?: boolean;
  matched_as?: string | null;
  match_how?: string[];
  similarity?: number | null;
}

export interface SubgraphEdge {
  id: string;
  kind: EdgeKind;
  source: string;
  target: string;
  label: string;
  /** The server's flag. Never recomputed here — see the module note. */
  crosses_branches: boolean;
  branches: string[];
  hop?: number;
  evidence?: string;
  source_chunk?: string;
}

export interface Subgraph {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
  branches: string[];
  /** Edges, not connections: one crossing is usually several edges. */
  crossing_edges: number;
}

export interface SourceVia {
  entity: string;
  type: string;
  hop: number;
}

export interface Source {
  source: number;
  cited: boolean;
  chunk_id: string;
  document: string;
  filename: string;
  index: number;
  page: number | null;
  text: string;
  score: number;
  legs: string[];
  ranks: Record<string, number>;
  hop: number | null;
  /** The node id this source is drawn as. How a citation finds its node. */
  node: string;
  ancestry: Ancestor[];
  location: string;
  via: SourceVia[];
}

export interface Rewrite {
  asked: string;
  standalone: string;
  changed: boolean;
  note: string;
}

export interface Tuning {
  legs: Legs;
  mode: Mode;
  top_k: number;
  hops: number;
  max_chunks: number;
}

export interface Traversal {
  hops: number;
  entities: { surface: string; type: string }[];
  matches: {
    surface: string;
    entity: string;
    node: string;
    scope: string;
    how: string[];
    similarity: number | null;
  }[];
  reached: number;
  edges: number;
}

export interface LegBalance {
  context: number;
  search: number;
  traversal: number;
  both: number;
  /** Chunks the lexical leg never returned. The demo's claim, as a number. */
  traversal_only: number;
}

export interface AskResponse {
  question: { asked: string; answered: string };
  rewrite: Rewrite;
  answer: string;
  grounded: boolean;
  citations: number[];
  uncited: number[];
  parameters: Tuning;
  sources: Source[];
  branches: string[];
  legs: LegBalance;
  traversal: Traversal | null;
  subgraph: Subgraph;
}

export interface StatusLevel {
  depth: number;
  name: string;
  confidence: number;
  folders: number;
}

export interface StatusResponse {
  server: string;
  database: string;
  ready: boolean;
  populating: boolean;
  nodes: Record<string, number>;
  entities: { total: number; bridges: number; folder_owned: number };
  levels: StatusLevel[];
  ingest_command: string;
}

export interface Exchange {
  question: string;
  answer: string;
}

export interface AskRequest {
  question: string;
  tuning: Tuning;
  /** Oldest first, each question in its **standalone** form — see `App`. */
  history: Exchange[];
  window: number;
}

/** A failure with the service's own words in it.
 *
 * The 503 path in `api.py` puts the reason in `detail` — "Neo4j: ...", "Embedding
 * failed: ..." — and each of those is recoverable by the operator. Showing
 * "Request failed" instead would hide the one thing worth knowing on stage. */
export class ServiceError extends Error {}

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ServiceError(
      `Der Dienst ist nicht erreichbar (${path}). Läuft \`just serve\`?`,
    );
  }
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined);
    throw new ServiceError(detail ?? `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export function ask(request: AskRequest, signal?: AbortSignal): Promise<AskResponse> {
  return post<AskResponse>("/api/ask", request, signal);
}

export async function status(): Promise<StatusResponse> {
  const response = await fetch("/api/status");
  if (!response.ok) throw new ServiceError(`${response.status} ${response.statusText}`);
  return (await response.json()) as StatusResponse;
}

/**
 * What the UI opens on: the CLI's defaults, unchanged.
 *
 * Two hops rather than one, and that was measured rather than assumed. At one hop
 * the target question draws a sparser, prettier picture and gets a visibly worse
 * answer — the model reports that no connection between the two projects can be
 * established from the sources. The second hop is what reaches the shared
 * reference architecture. Density is dealt with where it belongs: "Nur die
 * Übergänge" and the hop slider, both on screen, both turnable mid-sentence.
 */
export const DEFAULT_TUNING: Tuning = {
  legs: "both",
  mode: "hybrid",
  top_k: 10,
  hops: 2,
  max_chunks: 12,
};
