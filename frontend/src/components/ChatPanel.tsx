/**
 * The answer, and everything that joins it to the picture beside it.
 *
 * Three things here are load-bearing rather than decorative:
 *
 * **`[Quelle 3]` is a control, not text.** Hovering it lights the chunk node it was
 * written from; clicking it focuses that node's whole story — its folder chain and
 * the entities it led to. Without that, "the citations connect to the graph" is a
 * claim the operator has to make verbally, which is exactly what this screen exists
 * to avoid.
 *
 * **The leg each source came from is on its face.** `nur Graph` means the lexical
 * leg never returned that chunk. With the flat-RAG baseline cut from scope, that
 * badge is the most direct statement of the thesis anywhere in the UI.
 *
 * **A rewritten follow-up is shown, not applied silently.** The audience asked "und
 * 2023?"; what was retrieved was something else, and watching the substitution is
 * the moment.
 */

import type { AskResponse, Source } from "../api";
import { UNPLACED } from "../theme";

export interface Turn {
  id: number;
  asked: string;
  response: AskResponse;
}

interface Props {
  turns: Turn[];
  current: Turn | null;
  pending: string | null;
  error: string | null;
  selected: string | null;
  onSelectTurn: (id: number) => void;
  onSelectNode: (id: string | null) => void;
  onHoverNode: (id: string | null) => void;
  hues: Map<string, string>;
}

const LEG_LABELS: Record<string, string> = {
  search: "lexikalisch",
  traversal: "Traversierung",
};

function legBadges(source: Source) {
  const only = source.legs.length === 1 && source.legs[0] === "traversal";
  return (
    <>
      {only ? (
        <span className="badge badge-graph" title="Die lexikalische Suche hat diesen Abschnitt nie zurückgegeben — er kam allein über den Entitätengraphen.">
          nur Graph
        </span>
      ) : (
        source.legs.map((leg) => (
          <span key={leg} className="badge">
            {LEG_LABELS[leg] ?? leg}
          </span>
        ))
      )}
    </>
  );
}

/** The prose, with every `[Quelle N]` turned into a control over the picture. */
function Prose({
  text,
  sources,
  onSelectNode,
  onHoverNode,
  selected,
}: {
  text: string;
  sources: Source[];
  onSelectNode: (id: string | null) => void;
  onHoverNode: (id: string | null) => void;
  selected: string | null;
}) {
  const byNumber = new Map(sources.map((source) => [source.source, source]));
  // Split on the citation form `answer.py` instructs the model to use, and on the
  // one piece of markdown the chat tier emits unasked: it bolds proper nouns,
  // because that is how it writes prose everywhere else. Rendering the asterisks
  // literally makes the answer look like a debug dump of itself. Anything that is
  // neither falls through as text, including a citation pointing at a source that
  // does not exist — which the panel reports above rather than hides.
  const parts = text.split(/(\[Quelle\s+\d+\]|\*\*[^*]+\*\*)/g);
  return (
    <p className="prose">
      {parts.map((part, index) => {
        const bold = /^\*\*([^*]+)\*\*$/.exec(part);
        if (bold) return <strong key={index}>{bold[1]}</strong>;
        const match = /^\[Quelle\s+(\d+)\]$/.exec(part);
        if (!match) return <span key={index}>{part}</span>;
        const number = Number(match[1]);
        const source = byNumber.get(number);
        if (!source) {
          return (
            <span key={index} className="cite cite-missing" title="Diese Quelle gibt es nicht.">
              {part}
            </span>
          );
        }
        return (
          <button
            key={index}
            type="button"
            className={`cite${selected === source.node ? " cite-active" : ""}`}
            onMouseEnter={() => onHoverNode(source.node)}
            onMouseLeave={() => onHoverNode(null)}
            onClick={() => onSelectNode(selected === source.node ? null : source.node)}
          >
            {number}
          </button>
        );
      })}
    </p>
  );
}

function SourceCard({
  source,
  selected,
  onSelectNode,
  onHoverNode,
  hue,
}: {
  source: Source;
  selected: boolean;
  onSelectNode: (id: string | null) => void;
  onHoverNode: (id: string | null) => void;
  hue: string;
}) {
  return (
    <button
      type="button"
      className={`source${selected ? " source-active" : ""}`}
      style={{ borderLeftColor: hue }}
      onMouseEnter={() => onHoverNode(source.node)}
      onMouseLeave={() => onHoverNode(null)}
      onClick={() => onSelectNode(selected ? null : source.node)}
    >
      <div className="source-head">
        <span className="source-number">{source.source}</span>
        <span className="source-location">{source.location}</span>
        <span className="source-file dim">
          {source.filename}
          {source.page !== null ? ` · Seite ${source.page}` : ""}
        </span>
        {source.cited ? <span className="badge badge-cited">zitiert</span> : null}
        {legBadges(source)}
      </div>
      {source.via.length > 0 ? (
        <div className="source-via">
          über{" "}
          {source.via.map((via, index) => (
            <span key={`${via.entity}-${index}`}>
              {index > 0 ? ", " : ""}
              <em>{via.entity}</em> <span className="dim">(Hop {via.hop})</span>
            </span>
          ))}
        </div>
      ) : null}
      <div className="source-text">{source.text}</div>
    </button>
  );
}

export function ChatPanel({
  turns,
  current,
  pending,
  error,
  selected,
  onSelectTurn,
  onSelectNode,
  onHoverNode,
  hues,
}: Props) {
  return (
    <div className="chat">
      {turns.length > 1 ? (
        <div className="turn-strip">
          {turns.map((turn, index) => (
            <button
              key={turn.id}
              type="button"
              className={`turn-chip${current?.id === turn.id ? " turn-chip-active" : ""}`}
              onClick={() => onSelectTurn(turn.id)}
              title={turn.asked}
            >
              {index + 1}. {turn.asked}
            </button>
          ))}
        </div>
      ) : null}

      {error ? <div className="error">{error}</div> : null}
      {pending ? (
        <div className="pending">
          <span className="spinner" /> <span>{pending}</span>
        </div>
      ) : null}

      {current ? (
        <article className="answer">
          <h2 className="asked">{current.asked}</h2>
          {current.response.rewrite.changed ? (
            <div className="rewrite">
              <span className="rewrite-label">Rückfrage aufgelöst</span>
              <span className="rewrite-to">{current.response.rewrite.standalone}</span>
            </div>
          ) : null}

          {current.response.uncited.length > 0 ? (
            <div className="warning">
              Zitiert Quellen, die es nicht gibt: {current.response.uncited.join(", ")}
            </div>
          ) : null}
          {!current.response.grounded ? (
            <div className="warning">Die Antwort nennt keine Quelle.</div>
          ) : null}

          <Prose
            text={current.response.answer}
            sources={current.response.sources}
            onSelectNode={onSelectNode}
            onHoverNode={onHoverNode}
            selected={selected}
          />

          <h3 className="sources-head">
            Quellen
            <span className="dim">
              {" "}
              — {current.response.legs.traversal_only} von {current.response.legs.context} nur
              über den Graphen gefunden
            </span>
          </h3>
          <div className="sources">
            {current.response.sources.map((source) => (
              <SourceCard
                key={source.chunk_id}
                source={source}
                selected={selected === source.node}
                onSelectNode={onSelectNode}
                onHoverNode={onHoverNode}
                hue={hues.get(source.ancestry[0]?.name ?? "") ?? UNPLACED}
              />
            ))}
          </div>
        </article>
      ) : null}
    </div>
  );
}
