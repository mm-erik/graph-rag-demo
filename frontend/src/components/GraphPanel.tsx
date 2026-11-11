/**
 * The picture. The primary visual of the whole project, not a panel beside the chat.
 *
 * A force-directed layout, deliberately: the thing being drawn is a hierarchy with
 * an entity web laid over it, and a strictly hierarchical layout has nowhere to put
 * the edges that leave the tree — which are the only edges that matter here.
 *
 * Everything in the painting is in service of one sentence being readable from the
 * back of a room: *the answer came from two different clients, joined by something
 * neither folder contains.* Hence shape for kind, branch hue for fill, and gold —
 * thick, and the only moving thing on screen — for every edge the server flagged as
 * crossing. Motion is reserved for the same idea as the colour: the eye finds it
 * before it finds anything else, which is exactly the order the argument needs.
 */

import { forceCollide } from "d3-force";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

import type { Focus, RenderEdge, RenderGraph, RenderNode } from "../graph";
import { DETAIL_WIDTH, palette } from "../theme";

interface ForceGraphHandle {
  d3Force: {
    (name: string): { distance?: (fn: unknown) => void; strength?: (fn: unknown) => void } | undefined;
    (name: string, force: unknown): void;
  };
  zoom: (k: number, ms?: number) => void;
  centerAt: (x: number, y: number, ms?: number) => void;
}

interface Props {
  graph: RenderGraph;
  focus: Focus | null;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onHover: (node: RenderNode | null) => void;
  onHoverEdge: (edge: RenderEdge | null) => void;
  /** Nodes the camera should frame. Null frames everything. Driven by the
   * *selection*, never by hover — a view that lurched on every mouse move would
   * be unusable in front of an audience. */
  zoomTo: Set<string> | null;
  /** Presentation scale. 1 is a laptop; the stage control goes up from there. */
  scale: number;
}

function withAlpha(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

/** One shape per kind, so the four node types survive a projector that has lost
 * half its colour fidelity. Shape is the encoding that always arrives. */
function shape(ctx: CanvasRenderingContext2D, node: RenderNode, r: number): void {
  const { x = 0, y = 0 } = node;
  switch (node.kind) {
    case "folder": {
      // A tab along the top: a folder, at the size this is drawn at.
      roundRect(ctx, x - r, y - r * 0.78, r * 2, r * 1.56, r * 0.3);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - r, y - r * 0.78);
      ctx.lineTo(x - r * 0.2, y - r * 0.78);
      ctx.lineTo(x - r * 0.05, y - r * 1.12);
      ctx.lineTo(x - r * 0.85, y - r * 1.12);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      break;
    }
    case "document": {
      roundRect(ctx, x - r * 0.78, y - r, r * 1.56, r * 2, r * 0.18);
      ctx.fill();
      ctx.stroke();
      break;
    }
    case "chunk": {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      break;
    }
    case "entity": {
      ctx.beginPath();
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r, y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      break;
    }
  }
}

function label(node: RenderNode): string {
  if (node.kind === "chunk") return node.label;
  if (node.kind === "document") return node.label.replace(/\.pdf$/i, "");
  return node.label;
}

export function GraphPanel({
  graph,
  focus,
  selected,
  onSelect,
  onHover,
  onHoverEdge,
  zoomTo,
  scale,
}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const forceGraph = useRef<ForceGraphHandle | null>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // How far the drawing of a node reaches from its centre, label included. Measured
  // the way `nodeCanvasObject` draws it: same font sizes, same rule about which
  // nodes carry a label at rest. 0.3 per character is a workable stand-in for the
  // width of this typeface — the camera needs a bound, not a typographic truth.
  const fontOf = (node: RenderNode) =>
    (node.kind === "folder" && node.depth === 1 ? 13 : 11) * scale;
  const halfWidth = useCallback(
    (node: RenderNode) =>
      Math.max(
        node.radius * scale,
        node.keyLabel ? label(node).length * fontOf(node) * 0.3 : 0,
      ),
    [scale],
  );
  const labelHeight = useCallback(
    (node: RenderNode) => (node.keyLabel ? fontOf(node) * 2 : 0),
    [scale],
  );

  // The layout, and it is doing argumentative work rather than looking tidy.
  //
  // Containment is pulled tight and given full strength, so each client's folder
  // tree collapses into a compact cluster that reads as *one place in the
  // hierarchy*. Crossing edges are given the longest rest length in the graph and
  // the bridges they end at repel everything hardest, so the two clusters are held
  // apart and the gold is stretched between them with room to be seen. The shape
  // the audience ends up looking at — two tight trees, a span of gold across the
  // gap — is the sentence the demo is making, drawn.
  useEffect(() => {
    const instance = forceGraph.current;
    if (!instance) return;
    const link = instance.d3Force("link");
    link?.distance?.((edge: RenderEdge) => {
      if (edge.kind === "contains") return 40;
      if (edge.kind === "has_chunk") return 32;
      if (edge.crosses_branches) return 150;
      return edge.kind === "mentions" ? 72 : 105;
    });
    link?.strength?.((edge: RenderEdge) =>
      edge.kind === "contains" || edge.kind === "has_chunk" ? 1 : 0.14,
    );
    const charge = instance.d3Force("charge");
    charge?.strength?.((node: RenderNode) =>
      node.bridge ? -85 * node.radius : -46 * node.radius,
    );
    // Without this the five bridge entities settle into one gold heap with their
    // labels written over each other, which is precisely the part of the picture
    // that has to be readable.
    // Collision sized by what is *drawn*, label included, and scaled by the stage
    // control — unlike the link distances, which are not. At `groß` the labels are
    // drawn 1.6× larger into the same layout, so the room they need grows with
    // them; without this the bridge names end up written over each other, which is
    // precisely the part of the picture that has to be readable. Scaling the
    // distances instead would enlarge the whole drawing, the camera would zoom out
    // by exactly as much, and the control would do nothing at all.
    instance.d3Force(
      "collide",
      forceCollide<RenderNode>((node) =>
        Math.max(node.radius * scale * 1.7, halfWidth(node) * 0.8),
      ).strength(0.8),
    );
  }, [graph, scale, halfWidth]);

  const data = useMemo(() => ({ nodes: graph.nodes, links: graph.links }), [graph]);

  /**
   * Frame a set of nodes, keeping the right-hand strip clear.
   *
   * `zoomToFit` would do the fitting, but it centres on the whole canvas — and the
   * detail card sits over the right quarter of it, so selecting a node reliably
   * parked that node underneath the card describing it. The box is computed here
   * instead and the centre shifted by half the reserved strip.
   */
  const frame = useCallback(
    (ids: Set<string> | null) => {
      const instance = forceGraph.current;
      if (!instance) return;
      const framed = ids
        ? graph.nodes.filter((node) => ids.has(node.id))
        : graph.nodes;
      if (framed.length === 0) return;
      // The box is built around what is actually *drawn*, not around the node
      // positions. A label is wider than the dot it belongs to and is drawn larger
      // as the stage control goes up, so a box of positions plus a fixed margin
      // shears the outermost client's name off at exactly the scale meant for the
      // back of the room. Measuring instead of guessing makes the fit correct at
      // every scale, and keeps the margin small enough that the panel stays full.
      const left = Math.min(...framed.map((node) => (node.x ?? 0) - halfWidth(node)));
      const right = Math.max(...framed.map((node) => (node.x ?? 0) + halfWidth(node)));
      const top = Math.min(...framed.map((node) => (node.y ?? 0) - node.radius * scale * 1.6));
      const bottom = Math.max(
        ...framed.map((node) => (node.y ?? 0) + node.radius * scale * 1.6 + labelHeight(node)),
      );
      const padding = 36;
      const reserved = ids ? DETAIL_WIDTH : 0;
      const zoom = Math.min(
        Math.max(size.width - reserved, 200) / (right - left + padding * 2),
        size.height / (bottom - top + padding * 2),
        // A single selected node would otherwise fill the panel at absurd
        // magnification; the cap keeps its neighbourhood in shot.
        2.4,
      );
      instance.zoom(zoom, 600);
      instance.centerAt(
        (left + right) / 2 + reserved / (2 * zoom),
        (top + bottom) / 2,
        600,
      );
    },
    [graph, scale, size.width, size.height, halfWidth, labelHeight],
  );

  // Fit the whole graph when the data changes, and keep trying for a few seconds:
  // the simulation is re-heated by the layout effect above and by React rendering
  // new data, so one fit can land while the nodes are still travelling — leaving a
  // demo whose picture sits small in the corner of a large empty panel.
  //
  // `scale` is in the dependencies because the collide force grows with it, which
  // re-heats the simulation and spreads the graph: without a re-fit, turning the
  // stage control up pushed the outermost nodes off the edge of the panel.
  useEffect(() => {
    const timers = [400, 1200, 2600, 4500].map((delay) =>
      setTimeout(() => frame(null), delay),
    );
    return () => timers.forEach(clearTimeout);
  }, [data, scale, frame]);

  useEffect(() => {
    const timer = setTimeout(() => frame(zoomTo), 60);
    return () => clearTimeout(timer);
  }, [zoomTo, frame]);

  const lit = (id: string): boolean => !focus || focus.nodes.has(id);
  const litEdge = (edge: RenderEdge): boolean => !focus || focus.edges.has(edge.id);

  return (
    <div className="graph-panel" ref={container}>
      <ForceGraph2D
        ref={forceGraph as never}
        width={size.width}
        height={size.height}
        graphData={data}
        backgroundColor={palette.background}
        cooldownTicks={220}
        d3VelocityDecay={0.28}
        onEngineStop={() => frame(null)}
        onNodeClick={(node: RenderNode) =>
          onSelect(node.id === selected ? null : node.id)
        }
        onBackgroundClick={() => onSelect(null)}
        onNodeHover={(node: RenderNode | null) => onHover(node)}
        onLinkHover={(edge: RenderEdge | null) => onHoverEdge(edge)}
        linkLabel={() => ""}
        linkHoverPrecision={6}
        nodeRelSize={1}
        nodeVal={(node: RenderNode) => node.radius}
        nodeLabel={() => ""}
        linkColor={(edge: RenderEdge) =>
          litEdge(edge) ? edge.color : withAlpha(edge.color, 0.1)
        }
        linkWidth={(edge: RenderEdge) =>
          (litEdge(edge) ? edge.width : edge.width * 0.5) * scale
        }
        linkDirectionalArrowLength={(edge: RenderEdge) =>
          edge.kind === "relates" ? 7 * scale : 0
        }
        linkDirectionalArrowRelPos={0.98}
        // Crossings are bowed apart. Twenty straight gold lines between the same
        // handful of nodes overlap into one mass; bowed, they read as twenty.
        linkCurvature={(edge: RenderEdge) => (edge.crosses_branches ? 0.22 : 0)}
        // Only the crossings move. Motion is the strongest signal available and it
        // is spent entirely on the one thing the demo exists to show.
        linkDirectionalParticles={(edge: RenderEdge) =>
          edge.crosses_branches && litEdge(edge) ? 4 : 0
        }
        linkDirectionalParticleWidth={3.5 * scale}
        linkDirectionalParticleSpeed={0.006}
        linkDirectionalParticleColor={() => palette.crossing}
        nodeCanvasObject={(
          node: RenderNode,
          ctx: CanvasRenderingContext2D,
          zoom: number,
        ) => {
          const on = lit(node.id);
          const r = node.radius * scale;
          ctx.globalAlpha = on ? 1 : 0.16;

          if (node.id === selected) {
            ctx.beginPath();
            ctx.arc(node.x ?? 0, node.y ?? 0, r * 1.9, 0, Math.PI * 2);
            ctx.fillStyle = withAlpha(palette.text, 0.14);
            ctx.fill();
          }
          // A bridge wears a halo as well as the colour: on a washed-out projector
          // the ring survives when the hue does not.
          if (node.bridge) {
            ctx.beginPath();
            ctx.arc(node.x ?? 0, node.y ?? 0, r * 1.5, 0, Math.PI * 2);
            ctx.strokeStyle = withAlpha(palette.crossing, 0.45);
            ctx.lineWidth = 3 * scale;
            ctx.stroke();
          }

          ctx.fillStyle = node.color;
          ctx.strokeStyle =
            node.id === selected ? palette.text : withAlpha(palette.background, 0.9);
          ctx.lineWidth = 2 * scale;
          shape(ctx, node, r);

          // A chunk the answer actually cited gets a white core. The audience is
          // told "the answer quoted these"; this is where that is checkable.
          if (node.kind === "chunk" && node.cited) {
            ctx.beginPath();
            ctx.arc(node.x ?? 0, node.y ?? 0, r * 0.38, 0, Math.PI * 2);
            ctx.fillStyle = palette.background;
            ctx.fill();
          }

          const showing = node.keyLabel || zoom > 1.8 || node.id === selected;
          if (!showing || !on) {
            ctx.globalAlpha = 1;
            return;
          }
          const text = label(node);
          const fontSize = (node.kind === "folder" && node.depth === 1 ? 13 : 11) * scale;
          ctx.font = `${node.bridge || node.kind === "chunk" ? 700 : 500} ${fontSize}px Inter, system-ui, sans-serif`;
          const width = ctx.measureText(text).width;
          const y = (node.y ?? 0) + r + fontSize * 0.5 + 3 * scale;
          // A plate behind the text, because a label crossing an edge at a distance
          // is a label nobody reads.
          ctx.fillStyle = withAlpha(palette.background, 0.82);
          roundRect(
            ctx,
            (node.x ?? 0) - width / 2 - 4 * scale,
            y - fontSize * 0.72,
            width + 8 * scale,
            fontSize * 1.44,
            3 * scale,
          );
          ctx.fill();
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = node.bridge ? palette.crossing : palette.text;
          ctx.fillText(text, node.x ?? 0, y + fontSize * 0.02);
          ctx.globalAlpha = 1;
        }}
        nodePointerAreaPaint={(
          node: RenderNode,
          color: string,
          ctx: CanvasRenderingContext2D,
        ) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x ?? 0, node.y ?? 0, node.radius * scale * 1.4, 0, Math.PI * 2);
          ctx.fill();
        }}
      />
    </div>
  );
}
