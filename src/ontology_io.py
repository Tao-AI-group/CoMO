"""
ontology_io.py
--------------
Parse the COMBINI ontology (JSON-LD / OWL export) into a ground-truth graph
and expose the pieces the build-eval pipeline needs:

  - node table:        id -> {label, parents, defn, alt}
  - label<->id maps
  - subtree extraction (restrict to one branch, e.g. the intervention branch)
  - gold paths / gold parents (handles MULTIPLE inheritance)
  - skeleton vs. concept split (category nodes vs. everything to be placed)

The ontology hierarchy is treated as the GROUND TRUTH (expert-reviewed).
We strip the parent links from the concepts and ask an LLM to rebuild them;
this module is the "answer key" side.
"""
from __future__ import annotations
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

BASE = "https://github.com/Tao-AI-group/COMBINI#"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
SKOS_DEF = "http://www.w3.org/2004/02/skos/core#definition"
SKOS_ALT = "http://www.w3.org/2004/02/skos/core#altLabel"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"


@dataclass
class Node:
    id: str
    label: str
    parents: list[str] = field(default_factory=list)   # ids, in-graph only
    defn: str = ""
    alt: list[str] = field(default_factory=list)


class Ontology:
    def __init__(self, nodes: dict[str, Node]):
        self.nodes = nodes
        self.label2id = {n.label: nid for nid, n in nodes.items()}
        self.children: dict[str, list[str]] = defaultdict(list)
        for nid, n in nodes.items():
            for p in n.parents:
                if p in nodes:
                    self.children[p].append(nid)
        # stable ordering
        for p in self.children:
            self.children[p].sort(key=lambda c: nodes[c].label)

    # ---- lookups -------------------------------------------------------
    def lid(self, label: str) -> str:
        return self.label2id[label]

    def lab(self, nid: str) -> str:
        return self.nodes[nid].label

    def is_category(self, nid: str) -> bool:
        """A node that has children in the (full) graph = a skeleton category."""
        return len(self.children.get(nid, [])) > 0

    # ---- subtree -------------------------------------------------------
    def subtree(self, root_label: str, include_root: bool = False) -> set[str]:
        root = self.lid(root_label)
        seen, q = set(), deque([root])
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x)
            for c in self.children.get(x, []):
                q.append(c)
        if not include_root:
            seen.discard(root)
        return seen

    # ---- gold structure (multi-parent aware) ---------------------------
    def gold_parents(self, nid: str, scope: Optional[set[str]] = None) -> list[str]:
        ps = [p for p in self.nodes[nid].parents if p in self.nodes]
        if scope is not None:
            ps = [p for p in ps if p in scope or p == self._scope_root]
        return ps

    def gold_paths(self, nid: str, root: str) -> list[list[str]]:
        """All root->...->parent->nid paths (one per inheritance line)."""
        if nid == root:
            return [[root]]
        ps = [p for p in self.nodes[nid].parents if p in self.nodes]
        ps = [p for p in ps if self._reaches(p, root)]
        if not ps:
            return [[nid]]
        out = []
        for p in ps:
            for pre in self.gold_paths(p, root):
                out.append(pre + [nid])
        return out

    def _reaches(self, nid: str, root: str, _seen=None) -> bool:
        if nid == root:
            return True
        _seen = _seen or set()
        if nid in _seen:
            return False
        _seen.add(nid)
        return any(self._reaches(p, root, _seen)
                   for p in self.nodes[nid].parents if p in self.nodes)

    def ancestors(self, nid: str, root: str) -> set[str]:
        """Union of all category ancestors (excl. root, excl. self)."""
        anc = set()
        for path in self.gold_paths(nid, root):
            anc.update(path[:-1])         # drop self
        anc.discard(root)
        return anc

    _scope_root = None


def _load_jsonld(path: str) -> Ontology:
    """Fast hand-parser for the COMBINI JSON-LD dump (no rdflib needed)."""
    with open(path) as f:
        data = json.load(f)
    nodes: dict[str, Node] = {}
    for it in data:
        iid = it.get("@id", "")
        if not iid.startswith(BASE):
            continue
        if OWL_CLASS not in it.get("@type", []):
            continue
        nid = iid.replace(BASE, "")
        lab = it.get(RDFS_LABEL, [{}])
        lab = lab[0].get("@value", nid) if lab else nid
        parents = [p["@id"].replace(BASE, "")
                   for p in it.get(RDFS_SUBCLASS, []) if "@id" in p]
        defs = it.get(SKOS_DEF, [])
        defn = defs[0].get("@value", "") if defs else ""
        alts = [a.get("@value", "") for a in it.get(SKOS_ALT, [])]
        nodes[nid] = Node(nid, lab, parents, defn, alts)
    return Ontology(nodes)


def _load_rdf(path: str, fmt: Optional[str] = None,
              strip_ns: Optional[str] = None) -> Ontology:
    """
    Format-agnostic loader via rdflib. Reads ANY OWL serialization:
    RDF/XML (.owl/.rdf), Turtle (.ttl), OWL/XML, JSON-LD, N-Triples, ...

    The same RDF/OWL vocabulary URIs are used regardless of syntax; only the
    parsing is delegated to rdflib. Node ids are the full class URIs.

    Notes vs. the JSON-LD hand-parser:
      * Only NAMED superclasses are kept. `rdfs:subClassOf` pointing to a
        blank node (an anonymous class expression / owl:Restriction such as
        `hasPart some X`) is skipped -- those are axioms, not taxonomy edges.
      * label:      rdfs:label -> skos:prefLabel -> URI local name
      * definition: skos:definition -> rdfs:comment
      * altLabel:   skos:altLabel
    """
    import rdflib
    from rdflib import RDF, RDFS, OWL, URIRef
    from rdflib.namespace import SKOS

    if fmt is None:
        ext = path.rsplit(".", 1)[-1].lower()
        fmt = {"owl": "xml", "rdf": "xml", "xml": "xml",
               "ttl": "turtle", "n3": "n3", "nt": "nt",
               "jsonld": "json-ld", "json": "json-ld"}.get(ext, None)
    g = rdflib.Graph()
    g.parse(path, format=fmt)

    def local(uri: str) -> str:
        return uri.split("#")[-1].split("/")[-1]

    def short(uri: str) -> str:
        u = str(uri)
        if strip_ns and u.startswith(strip_ns):
            return u[len(strip_ns):]
        return u

    def first_str(subj, *preds):
        for p in preds:
            v = g.value(subj, p)
            if v is not None:
                return str(v)
        return ""

    named = [s for s in g.subjects(RDF.type, OWL.Class)
             if isinstance(s, URIRef)]
    nodes: dict[str, Node] = {}
    for s in named:
        sid = short(s)
        lab = first_str(s, RDFS.label, SKOS.prefLabel) or local(str(s))
        parents = [short(o) for o in g.objects(s, RDFS.subClassOf)
                   if isinstance(o, URIRef)]
        defn = first_str(s, SKOS.definition, RDFS.comment)
        alts = [str(o) for o in g.objects(s, SKOS.altLabel)]
        nodes[sid] = Node(sid, lab, parents, defn, alts)
    return Ontology(nodes)


def load_ontology(path: str, fmt: Optional[str] = None,
                  strip_ns: Optional[str] = None) -> Ontology:
    """
    Dispatch by file type. Same Ontology object either way, so build_eval.py
    is unchanged.

      *.json / *.jsonld  -> fast hand-parser (COMBINI dump, no rdflib)
      *.owl/.rdf/.ttl/... -> rdflib loader (any OWL serialization)

    Pass fmt=... to force an rdflib format (e.g. fmt="turtle").
    Pass strip_ns=... to shorten ids by removing a namespace prefix, e.g.
    strip_ns="https://github.com/Tao-AI-group/COMBINI#".
    """
    ext = path.rsplit(".", 1)[-1].lower()
    if fmt is None and ext in ("json", "jsonld"):
        return _load_jsonld(path)
    return _load_rdf(path, fmt=fmt, strip_ns=strip_ns)


if __name__ == "__main__":
    import sys
    onto = load_ontology(sys.argv[1] if len(sys.argv) > 1 else "ontology.json")
    sub = onto.subtree("Complementary_Medicine_Intervention")
    print(f"nodes in graph: {len(onto.nodes)}")
    print(f"intervention subtree: {len(sub)}")
    cats = [n for n in sub if onto.is_category(n)]
    print(f"  category (skeleton) nodes: {len(cats)}")
    print(f"  leaf concepts:            {len(sub) - len(cats)}")