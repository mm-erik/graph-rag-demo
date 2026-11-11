/**
 * The payload as a drawable model: colours, sizes, what is dimmed, what is dropped.
 *
 * Everything here is derived *presentation*. The one thing it never derives is
 * `crosses_branches`, which arrives decided — see `api.ts`.
 *
 * Two problems this layer exists to solve, both of them about a projector rather
 * than about correctness:
 *
 * **Density.** At `hops: 2` the real demo question comes back with 96 nodes and 275
 * edges, 180 of them entity-to-entity relations. Drawn in full they are a hairball,
 * and a hairball makes no argument. `prune` keeps the relations that are *doing the
 * work* and throws the rest away, with the count of what it dropped reported so the
 * picture never silently claims to be everything.
 *
 * **Tracing.** "Which chunks were retrieved, and which entities did they lead to"
 * has to be answerable node by node from the back of a room. `focus` answers it for
 * one node: its own neighbourhood, plus — for a chunk — the whole folder chain above
 * it, because a chunk's place in the hierarchy is half of what it is evidence of.
 */

import type { Subgraph, SubgraphEdge, SubgraphNode } from "./api";
import { UNPLACED, branchHues, edgeColors, palette } from "./theme";

export interface RenderNode extends SubgraphNode {
  color: string;
  radius: number;
  /** Drawn at every zoom level. The rest get a label when zoomed in or lit. */
  keyLabel: boolean;
  // Written by the force simulation.
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

export interface RenderEdge extends Omit<SubgraphEdge, "source" | "target"> {
  /** `string` until the simulation swaps in the node object it resolves to. */
  source: string | RenderNode;
  target: string | RenderNode;
  color: string;
  width: number;
}

export interface RenderGraph {
  nodes: RenderNode[];
  links: RenderEdge[];
  /** Branch name → hue, so the legend and the chat panel agree with the picture. */
  hues: Map<string, string>;
  /** What `prune` dropped, reported rather than hidden: relations, and the
   * entities that were left with nothing to hang from once they went. */
  hidden: { edges: number; nodes: number };
}

export function endpointId(end: string | RenderNode): string {
  return typeof end === "string" ? end : end.id;
}

export function hueMap(branches: string[]): Map<string, string> {
  // Sorted, so a branch keeps its colour between two runs of the same question —
  // a demo whose picture recolours itself on a re-ask cannot be rehearsed.
  return new Map(
    [...branches].sort().map((branch, index) => [
      branch,
      branchHues[index % branchHues.length],
    ]),
  );
}

function nodeColor(node: SubgraphNode, hues: Map<string, string>): string {
  // Gold first: a bridge is a bridge before it is an entity, and this is the one
  // place the reservation in `theme.ts` is enforced.
  if (node.bridge) return palette.crossing;
  if (node.branches.length === 1) {
    return hues.get(node.branches[0]) ?? UNPLACED;
  }
  // An entity no drawn chunk mentions sits nowhere in particular. Neutral, not
  // guessed at — the same restraint the server shows by leaving its edges unflagged.
  return node.kind === "entity" ? palette.entity : UNPLACED;
}

function radius(node: SubgraphNode): number {
  switch (node.kind) {
    case "folder":
      // The top-level folder is the client: the largest thing in the picture, so
      // the two trees are separable before any label is read.
      return node.depth === 1 ? 17 : 11;
    case "document":
      return 9;
    case "chunk":
      return node.cited ? 12 : 9;
    case "entity":
      if (node.bridge) return 15;
      return node.hop === 0 ? 10 : 7;
  }
}

function keyLabel(node: SubgraphNode): boolean {
  // What stays legible when the whole subgraph is on screen: the tree, every
  // numbered source, and the entities the question named or the walk bridged.
  if (node.kind === "entity") return node.bridge || node.hop === 0;
  return true;
}

function edgeColor(edge: SubgraphEdge): string {
  return edge.crosses_branches ? palette.crossing : edgeColors[edge.kind];
}

function edgeWidth(edge: SubgraphEdge): number {
  if (edge.crosses_branches) return 4;
  return edge.kind === "relates" || edge.kind === "mentions" ? 1.4 : 2;
}

/**
 * The relations worth drawing, when the picture has to read from a distance.
 *
 * The rule is **named**: a relation stays if it crosses a branch, or if both of its
 * ends are entities a drawn chunk actually mentions. What goes is the fan running
 * outward from the walk's frontier to entities nothing on screen names — on the
 * target question some sixty edges ending in unlabelled dots, which cost exactly as
 * much room as the crossing they surround and say nothing an audience can check.
 *
 * Drawn by "both ends named" rather than "crosses" alone, because the ticket asks
 * that the chunks retrieved and the entities they led to be traceable node by node,
 * and the hop-by-hop path between those entities is relation edges. Keeping only
 * the crossings would colour the payoff correctly and delete the route to it.
 *
 * Entities left with no edge go with them, except the ones the question itself
 * named: a question term missing from its own picture is a hole, not a tidier view.
 *
 * Nothing the answer was written from is ever pruned — chunks, their documents and
 * their folders are structural and stay regardless. `alles` turns this off, and the
 * count of what was dropped is on screen either way.
 */
export function prune(
  nodes: SubgraphNode[],
  edges: SubgraphEdge[],
): {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
  hidden: { edges: number; nodes: number };
} {
  const named = new Set(
    edges.filter((edge) => edge.kind === "mentions").map((edge) => edge.target),
  );
  const kept = edges.filter(
    (edge) =>
      edge.kind !== "relates" ||
      edge.crosses_branches ||
      (named.has(edge.source) && named.has(edge.target)),
  );
  const touched = new Set<string>();
  for (const edge of kept) {
    touched.add(edge.source);
    touched.add(edge.target);
  }
  const keptNodes = nodes.filter(
    (node) => node.kind !== "entity" || touched.has(node.id) || node.hop === 0,
  );
  const alive = new Set(keptNodes.map((node) => node.id));
  return {
    nodes: keptNodes,
    edges: kept.filter((edge) => alive.has(edge.source) && alive.has(edge.target)),
    hidden: {
      edges: edges.length - kept.length,
      nodes: nodes.length - keptNodes.length,
    },
  };
}

export function buildGraph(subgraph: Subgraph, dense: boolean): RenderGraph {
  const { nodes, edges, hidden } = dense
    ? { nodes: subgraph.nodes, edges: subgraph.edges, hidden: { edges: 0, nodes: 0 } }
    : prune(subgraph.nodes, subgraph.edges);
  const hues = hueMap(subgraph.branches);
  return {
    // Copied, because the force simulation writes positions onto these objects and
    // replaces each link's endpoints with the node they resolve to. Handing it the
    // response's own objects would mutate the turn we still render the answer from.
    nodes: nodes.map((node) => ({
      ...node,
      color: nodeColor(node, hues),
      radius: radius(node),
      keyLabel: keyLabel(node),
    })),
    links: edges.map((edge) => ({
      ...edge,
      color: edgeColor(edge),
      width: edgeWidth(edge),
    })),
    hues,
    hidden,
  };
}

export interface Focus {
  nodes: Set<string>;
  edges: Set<string>;
}

/**
 * One node's story: what it touches, and — for a chunk — where it lives.
 *
 * The folder chain is followed all the way up rather than one hop, because the
 * question a lit chunk has to answer is "which client, which year, which project",
 * and that is three hops of containment above it. Entity neighbours are one hop:
 * two would relight most of the graph and stop being a focus at all.
 */
export function focusOn(graph: RenderGraph, id: string): Focus {
  const nodes = new Set<string>([id]);
  const edges = new Set<string>();
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));

  for (const link of graph.links) {
    const from = endpointId(link.source);
    const to = endpointId(link.target);
    if (from === id || to === id) {
      edges.add(link.id);
      nodes.add(from);
      nodes.add(to);
    }
  }

  // Upwards through containment, from whatever structural node we started at.
  const parents = new Map<string, { parent: string; edge: string }>();
  for (const link of graph.links) {
    if (link.kind === "contains" || link.kind === "has_chunk") {
      parents.set(endpointId(link.target), {
        parent: endpointId(link.source),
        edge: link.id,
      });
    }
  }
  // An entity has no place in the hierarchy by design, so there is nothing to climb.
  let current: string | undefined = byId.get(id)?.kind === "entity" ? undefined : id;
  const climbed = new Set<string>();
  while (current && !climbed.has(current)) {
    climbed.add(current);
    const step = parents.get(current);
    if (!step) break;
    nodes.add(step.parent);
    edges.add(step.edge);
    current = step.parent;
  }
  return { nodes, edges };
}

/** Every crossing edge and both of its ends — the "show me the payoff" view. */
export function focusOnCrossings(graph: RenderGraph): Focus {
  const nodes = new Set<string>();
  const edges = new Set<string>();
  for (const link of graph.links) {
    if (!link.crosses_branches) continue;
    edges.add(link.id);
    nodes.add(endpointId(link.source));
    nodes.add(endpointId(link.target));
  }
  return { nodes, edges };
}
