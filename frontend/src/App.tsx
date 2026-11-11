/**
 * The screen: an answer on the left, the subgraph that produced it on the right,
 * with comparable weight. The split is the ticket's central instruction and the
 * reason the layout is a two-column grid rather than a chat app with a diagram
 * tucked into a corner.
 *
 * State worth knowing about:
 *
 * **Turns are kept whole.** Each answer keeps its own subgraph, so clicking back
 * through the conversation redraws the picture that question actually produced
 * rather than re-asking it. A re-ask would cost a model call and might answer
 * differently, on stage, in front of the thing the audience just looked at.
 *
 * **History is sent in its standalone form.** `rewrite.standalone`, never what was
 * typed — a history of shorthand degrades one turn at a time, which reads as the
 * model getting worse rather than as a bug.
 *
 * **Focus is derived, never stored.** The lit set is a function of the selection
 * and the current graph, so a selection that survives into a turn where that node
 * does not exist simply lights nothing.
 */

import type React from "react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  DEFAULT_TUNING,
  ServiceError,
  type AskResponse,
  type Exchange,
  type StatusResponse,
  type Tuning,
  ask,
  status as fetchStatus,
} from "./api";
import { ChatPanel, type Turn } from "./components/ChatPanel";
import { Controls } from "./components/Controls";
import { GraphPanel } from "./components/GraphPanel";
import { Legend, NodeDetail, ProofLine } from "./components/GraphOverlays";
import { EdgeDetail } from "./components/EdgeDetail";
import { buildGraph, focusOn, focusOnCrossings, type RenderEdge } from "./graph";
import { DETAIL_WIDTH } from "./theme";

/** The question the whole project is pointed at. Offered as a button because the
 * one thing that must not go wrong on stage is typing it. */
const DEMO_QUESTIONS = [
  "Wer leitet das Projekt Atlas bei der Nordlicht Energie AG?",
  "Was verbindet das Projekt Atlas von Nordlicht Energie AG mit dem von Brandt und Soehne Maschinenbau?",
];

export function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tuning, setTuning] = useState<Tuning>(DEFAULT_TUNING);
  const [window_, setWindow] = useState(4);
  const [dense, setDense] = useState(false);
  const [scale, setScale] = useState(1);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<RenderEdge | null>(null);
  const [crossingsOnly, setCrossingsOnly] = useState(false);
  const [service, setService] = useState<StatusResponse | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchStatus()
      .then(setService)
      .catch(() => setService(null));
  }, []);

  const current = useMemo(
    () => turns.find((turn) => turn.id === currentId) ?? null,
    [turns, currentId],
  );

  const graph = useMemo(
    () =>
      current
        ? buildGraph(current.response.subgraph, dense)
        : {
            nodes: [],
            links: [],
            hues: new Map<string, string>(),
            hidden: { edges: 0, nodes: 0 },
          },
    [current, dense],
  );

  // Hover wins over selection, in the picture *and* in the card, so that the two
  // never disagree about which node is being talked about. Letting the lit set
  // follow the mouse while the card stayed on the selection is a small
  // inconsistency on a laptop and an unexplainable one on a projector.
  const hoverFocus = useMemo(
    () => (hovered ? focusOn(graph, hovered) : null),
    [graph, hovered],
  );
  const selectedFocus = useMemo(
    () => (selected ? focusOn(graph, selected) : null),
    [graph, selected],
  );
  const crossingFocus = useMemo(
    () => (crossingsOnly ? focusOnCrossings(graph) : null),
    [graph, crossingsOnly],
  );
  const focus = hoverFocus ?? selectedFocus ?? crossingFocus;

  // What the camera frames. Never the hover: a view that lurched on every mouse
  // move would be unusable in front of an audience.
  const zoomTo = (selectedFocus ?? crossingFocus)?.nodes ?? null;

  const lit = hovered ?? selected;
  const detail = useMemo(
    () => graph.nodes.find((node) => node.id === lit) ?? null,
    [graph, lit],
  );

  async function submit(text: string) {
    const asked = text.trim();
    if (!asked || pending) return;
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setPending(asked);
    setError(null);
    setQuestion("");
    // Oldest first, and each question in the form retrieval actually ran on.
    const history: Exchange[] = turns.map((turn) => ({
      question: turn.response.rewrite.standalone,
      answer: turn.response.answer,
    }));
    try {
      const response: AskResponse = await ask(
        { question: asked, tuning, history, window: window_ },
        controller.signal,
      );
      const turn: Turn = { id: Date.now(), asked, response };
      setTurns((previous) => [...previous, turn]);
      setCurrentId(turn.id);
      setSelected(null);
      setHovered(null);
    } catch (failure) {
      if (controller.signal.aborted) return;
      setError(
        failure instanceof ServiceError
          ? failure.message
          : `Unerwarteter Fehler: ${String(failure)}`,
      );
    } finally {
      if (!controller.signal.aborted) setPending(null);
    }
  }

  return (
    // `--stage` scales what the room has to read — the picture, the answer, the
    // legend — and deliberately not the control strip. Scaling the chrome too, which
    // is what an outer `font-size` does, grows the header until it has eaten a third
    // of the canvas and the graph has *shrunk* to fit what is left.
    <div
      className="app"
      style={
        { "--stage": scale, "--detail-width": `${DETAIL_WIDTH}px` } as React.CSSProperties
      }
    >
      <header className="header">
        <div className="brand">
          <span className="brand-title">Graph RAG</span>
          <span className="brand-sub">Antwort und Traversierung</span>
        </div>
        <Controls
          tuning={tuning}
          onTuning={setTuning}
          window={window_}
          onWindow={setWindow}
          dense={dense}
          onDense={setDense}
          hidden={graph.hidden}
          scale={scale}
          onScale={setScale}
          busy={pending !== null}
        />
        <div className="service" title={service?.ingest_command}>
          {service ? (
            <>
              <span className={service.ready ? "dot dot-ok" : "dot dot-bad"} />
              {service.nodes.Chunk ?? 0} Abschnitte · {service.entities.total} Entitäten ·{" "}
              {service.entities.bridges} Brücken
            </>
          ) : (
            <>
              <span className="dot dot-bad" /> Dienst nicht erreichbar
            </>
          )}
        </div>
      </header>

      <main className="split">
        <section className="pane pane-chat">
          <ChatPanel
            turns={turns}
            current={current}
            pending={pending}
            error={error}
            selected={selected}
            onSelectTurn={(id) => {
              setCurrentId(id);
              setSelected(null);
            }}
            onSelectNode={setSelected}
            onHoverNode={setHovered}
            hues={graph.hues}
          />
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              void submit(question);
            }}
          >
            <input
              value={question}
              placeholder="Frage stellen — Rückfragen werden aufgelöst"
              onChange={(event) => setQuestion(event.target.value)}
              autoFocus
            />
            <button type="submit" disabled={pending !== null || !question.trim()}>
              Fragen
            </button>
          </form>
          <div className="suggestions">
            {DEMO_QUESTIONS.map((suggestion, index) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => void submit(suggestion)}
                disabled={pending !== null}
                title={suggestion}
              >
                {index === DEMO_QUESTIONS.length - 1 ? "Zielfrage" : `Frage ${index + 1}`}
              </button>
            ))}
            {turns.length > 0 ? (
              <button
                type="button"
                className="ghost"
                onClick={() => {
                  setTurns([]);
                  setCurrentId(null);
                  setSelected(null);
                }}
              >
                Gespräch zurücksetzen
              </button>
            ) : null}
          </div>
        </section>

        <section className="pane pane-graph">
          {current ? (
            <ProofLine response={current.response}>
              <button
                type="button"
                className={`crossings${crossingsOnly ? " on" : ""}`}
                onClick={() => setCrossingsOnly((on) => !on)}
                title="Alles ausblenden außer den Kanten, die von einem Kunden zum anderen führen."
              >
                Nur die Übergänge
              </button>
            </ProofLine>
          ) : null}
          <div className="canvas">
            {current ? (
              <GraphPanel
                graph={graph}
                focus={focus}
                selected={selected}
                onSelect={setSelected}
                onHover={(node) => setHovered(node?.id ?? null)}
                onHoverEdge={setHoveredEdge}
                zoomTo={zoomTo}
                scale={scale}
              />
            ) : (
              <div className="empty">
                <p>
                  Stellen Sie eine Frage. Hier erscheint der Teilgraph, aus dem die Antwort
                  entstanden ist — die gefundenen Abschnitte, die Entitäten, zu denen sie
                  geführt haben, und <strong>in Gold jede Kante, die den Hierarchiebaum
                  verlässt</strong>.
                </p>
              </div>
            )}
            {detail ? <NodeDetail node={detail} onClose={() => setSelected(null)} /> : null}
            {!detail && hoveredEdge ? <EdgeDetail edge={hoveredEdge} /> : null}
          </div>
          {/* Below the canvas rather than floating over it. As an overlay the legend
              sat on top of whichever part of the graph the layout happened to put
              there, and at presentation scale it covered a quarter of it — hiding
              the picture in order to explain it. */}
          {current ? <Legend hues={graph.hues} /> : null}
        </section>
      </main>
    </div>
  );
}
