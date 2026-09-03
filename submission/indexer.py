"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM and BM25 scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision.

Postings are stored as a structure-of-arrays, CSR-style layout: one
`post_doc`/`post_tf` pair of numpy arrays shared across all terms, with
`post_off[term_id:term_id+2]` giving the slice for that term. This lets
scorers (`bm25.py`, `boolean_vsm.py`) do vectorized numpy work
(`acc[docs] += ...`) directly on postings slices instead of decoding a
Python-level structure per query.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one — so whatever this index needs at query time must
round-trip through `save()`/`load()` below, not just live as Python
attributes. The on-disk byte size of what `save()` writes is graded
directly (smaller, relative to the class median, scores better).
"""
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple
import array
import json
import os
import zlib
import numpy as np
from nltk.stem import PorterStemmer

from submission import codec, stoplists

try:
    # Optional C++ extension built by submission/setup.py at image-build
    # time. It replaces the per-token Python work in build() with an
    # equivalent C++ scan; see _tokenizer.pyx. Absent on a machine with no
    # compiler, in which case _build_stream_python() below does the same
    # job and produces the same index — tests/test_fast_tokenizer.py pins
    # that the two agree.
    from submission import _tokenizer as _fast_tokenizer
except ImportError:  # pragma: no cover - exercised only where unbuilt
    _fast_tokenizer = None

_MAGIC = b"A1IX"
_INDEX_FORMAT = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()

# Byte-level equivalent of `str.lower()` then `_TOKEN_RE.findall()`: fold
# A-Z onto a-z, keep a-z0-9, and turn every other byte into a space so that
# `bytes.split()` cuts exactly where the regex would have. Doing it this way
# moves the whole scan into two C calls (`translate`, `split`) instead of the
# regex engine, which is ~4x faster over the 29M tokens of the full corpus
# (17.4s -> 1.4s measured) and is what makes the build-time budget reachable.
#
# Multi-byte UTF-8 sequences become spaces, exactly as the regex treated them
# (no non-ASCII character is in [a-z0-9]), so the token stream is unchanged
# for any realistic input; tests/test_retrieval_components.py pins this.
_TRANSLATE_TABLE = bytes(
    (c + 32) if 65 <= c <= 90 else (c if (48 <= c <= 57 or 97 <= c <= 122) else 32)
    for c in range(256)
)

# Same, but '-' survives so that split() hands back whole hyphenated runs
# and compound_tokens() can decide what to do with them. Used only when the
# `compounds` tokenizer option is on.
_HYPHEN = 0x2D
_TRANSLATE_TABLE_HYPHEN = bytes(
    (c + 32) if 65 <= c <= 90
    else (c if (48 <= c <= 57 or 97 <= c <= 122 or c == _HYPHEN) else 32)
    for c in range(256)
)

# --- Tokenizer configuration -------------------------------------------
# These are the settings that ship. `scripts/tune.py` overrides them via
# configure_tokenizer() to compare alternatives under cross-validation; the
# winner is then written back here as the default. The active setting is
# persisted into the index by save() and restored by load(), so a fresh
# process can never score against an index built with a different
# tokenizer (which would silently wreck retrieval rather than crash).
DEFAULT_STOPLIST = "minimal"
DEFAULT_STEMMER = "porter"
# 1, not 2: keeping single-character tokens preserves the trailing token in
# "SARS-CoV-2" and similar domain vocabulary. Selected by scripts/tune.py
# --stage mintok, where it won all 5 folds (+0.0062 nDCG@10) with a tighter
# fold spread than dropping them, for 3.3% more postings.
DEFAULT_MIN_TOKEN_LEN = 1
# Also index the glued form of hyphenated runs ("sars-cov-2" -> "sarscov2"
# alongside "sars"/"cov"/"2"). Selected by scripts/tune.py --stage compound.
DEFAULT_COMPOUNDS = False

_STOPLIST_NAME = DEFAULT_STOPLIST
_STEMMER_NAME = DEFAULT_STEMMER
_MIN_TOKEN_LEN = DEFAULT_MIN_TOKEN_LEN
_COMPOUNDS = DEFAULT_COMPOUNDS
_STOPWORDS = stoplists.BY_NAME[DEFAULT_STOPLIST]


def configure_tokenizer(stoplist: str = DEFAULT_STOPLIST,
                        stemmer: str = DEFAULT_STEMMER,
                        min_token_len: int = DEFAULT_MIN_TOKEN_LEN,
                        compounds: bool = DEFAULT_COMPOUNDS) -> None:
    """Set the tokenization scheme. Used by scripts/tune.py to evaluate
    alternatives; the shipped configuration is the module defaults above.

    stoplist:      "minimal" | "standard" | "none"  (see submission/stoplists.py)
    stemmer:       "porter" | "light" | "none"   ("light" = inflectional only)
    compounds:     also emit the glued form of hyphenated runs
    min_token_len: shortest token kept (2 drops single characters, so
                   "SARS-CoV-2" keeps "sars"/"cov" but not "2")
    """
    global _STOPLIST_NAME, _STEMMER_NAME, _MIN_TOKEN_LEN, _COMPOUNDS, _STOPWORDS
    if stoplist not in stoplists.BY_NAME:
        raise ValueError(f"unknown stoplist {stoplist!r}")
    if stemmer not in {"porter", "light", "none"}:
        raise ValueError(f"unknown stemmer {stemmer!r}")
    _STOPLIST_NAME = stoplist
    _STEMMER_NAME = stemmer
    _MIN_TOKEN_LEN = min_token_len
    _COMPOUNDS = bool(compounds)
    _STOPWORDS = stoplists.BY_NAME[stoplist]
    _stem.cache_clear()


def tokenizer_config() -> dict:
    """The active tokenizer settings, as persisted alongside the index."""
    return {
        "stoplist": _STOPLIST_NAME,
        "stemmer": _STEMMER_NAME,
        "min_token_len": _MIN_TOKEN_LEN,
        "compounds": _COMPOUNDS,
    }


def _light_stem(token: str) -> str:
    """Inflectional-only stemming: plurals and the regular verb endings.

    Porter is a *derivational* stemmer — it conflates "organisation" with
    "organ" and "immunity" with "immune". On general English that mostly
    helps; on a scientific collection it destroys precisely the technical
    vocabulary that carries the signal, because the derivational forms are
    distinct concepts rather than variants of one. This is the Krovetz
    argument, and this is the cheap version of it: strip the inflections
    every English speaker agrees are the same word, and stop there.

    It is also ~35x faster than NLTK's Porter over the corpus vocabulary
    (0.10s vs 3.6s), which is a real efficiency gain, but the reason to
    consider it is retrieval quality — scripts/tune.py decides.
    """
    n = len(token)
    if n < 4 or token.isdigit():
        return token
    if token.endswith("ies") and n > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("ss"):
        return token
    if token.endswith("s") and not token.endswith("us") and n > 3:
        return token[:-1]
    if token.endswith("ing") and n > 5:
        return token[:-3]
    if token.endswith("ed") and n > 4:
        return token[:-2]
    return token


@lru_cache(maxsize=300_000)
def _stem(token: str) -> str:
    """Cache stemming because corpus vocabularies repeat tokens heavily."""
    if _STEMMER_NAME == "none":
        return token
    if _STEMMER_NAME == "light":
        return _light_stem(token)
    return _STEMMER.stem(token)


def split_bytes(text: str) -> List[bytes]:
    """Lowercase and split `text` into raw `[a-z0-9]+` runs, as bytes.

    No stopword or length filtering happens here — build() applies those
    once per distinct vocabulary entry rather than once per occurrence, so
    this stays a pure two-C-call scan. Query-time tokenization goes through
    the same function, which is what guarantees the query tokenizer and the
    index tokenizer can never drift apart.
    """
    if _COMPOUNDS:
        return _split_compounds(text)
    return text.encode("utf-8", "ignore").translate(_TRANSLATE_TABLE).split()


def _split_compounds(text: str) -> List[bytes]:
    """Split, but also emit the glued form of every hyphenated run.

    "sars-cov-2" becomes ["sars", "cov", "2", "sarscov2"]. The parts are
    kept as well as the join, so a query saying "SARS CoV 2" still matches
    on the parts while one saying "SARS-CoV-2" additionally gets an exact
    hit on a rare, highly discriminative term. This corpus is full of such
    forms — virus and variant names, gene and protein symbols, chemical
    names — and they are exactly the terms whose IDF is highest.
    """
    out: List[bytes] = []
    for chunk in text.encode("utf-8", "ignore").translate(
            _TRANSLATE_TABLE_HYPHEN).split():
        if _HYPHEN in chunk:
            parts = [p for p in chunk.split(b"-") if p]
            if not parts:
                continue
            out.extend(parts)
            if len(parts) > 1:
                out.append(b"".join(parts))
        else:
            out.append(chunk)
    return out


def raw_tokens(text: str) -> List[str]:
    """Lowercase, split, drop stopwords and too-short tokens — but do not
    stem. Split out from tokenize() so that build() can stem once per
    distinct vocabulary entry instead of once per token occurrence.
    """
    min_len = _MIN_TOKEN_LEN
    stop = _STOPWORDS
    out = []
    for raw in split_bytes(text):
        if len(raw) < min_len:
            continue
        token = raw.decode("ascii")
        if token not in stop:
            out.append(token)
    return out


def tokenize(text: str) -> List[str]:
    """Lowercase, remove ordinary English stopwords, and stem.

    This deliberately has no corpus- or topic-specific vocabulary rules, so
    the indexer generalizes to any English collection in the assignment.
    """
    return [_stem(token) for token in raw_tokens(text)]


def _stream_python(corpus, doc_id_map, raw_lens):
    """Streaming pass in pure Python.

    Returns (raw_terms, all_raw): the vocabulary as a list of bytes indexed
    by raw ordinal, and one int32 ordinal per token occurrence in document
    order. Appends to `doc_id_map` and `raw_lens` as it goes.
    """
    raw_vocab: Dict[bytes, int] = {}
    raw_get = raw_vocab.get
    raw_chunks: List[array.array] = []

    for doc_id, text in corpus:
        doc_id_map.append(doc_id)
        tokens = split_bytes(text)
        raw_lens.append(len(tokens))
        if not tokens:
            continue

        # map() in C is much faster than a Python-level loop; only the
        # tokens genuinely new to the vocabulary need the slow path.
        ordinals = list(map(raw_get, tokens))
        if None in ordinals:
            for i, ordinal in enumerate(ordinals):
                if ordinal is None:
                    token = tokens[i]
                    ordinal = raw_vocab.get(token)
                    if ordinal is None:
                        ordinal = len(raw_vocab)
                        raw_vocab[token] = ordinal
                    ordinals[i] = ordinal
        raw_chunks.append(array.array("i", ordinals))

    raw_terms = list(raw_vocab)                # insertion order == ordinal
    if raw_chunks:
        all_raw = np.concatenate([np.frombuffer(c, dtype=np.int32)
                                  for c in raw_chunks])
        raw_chunks.clear()                     # release the per-doc arrays
    else:
        all_raw = np.zeros(0, dtype=np.int32)
    return raw_terms, all_raw


def _stream_fast(corpus, doc_id_map, raw_lens):
    """Streaming pass through the C++ extension. Same contract as
    _stream_python; tests/test_fast_tokenizer.py asserts they agree."""
    builder = _fast_tokenizer.VocabBuilder()
    add = builder.add
    for doc_id, text in corpus:
        doc_id_map.append(doc_id)
        raw_lens.append(add(text.encode("utf-8", "ignore")))
    return builder.finish()


class InvertedIndex:
    """Structure-of-arrays inverted index.

    Postings for all terms live in two flat numpy arrays, `post_doc` and
    `post_tf`, sorted by (term_id, doc_id). `post_off[t]:post_off[t + 1]`
    is the slice of those arrays belonging to term ordinal `t`. This is
    the numpy analogue of a CSR sparse matrix, and it's what lets scorers
    do `acc[docs] += weights` instead of a per-posting Python loop.
    """

    def __init__(self):
        self.doc_id_map: List[str] = []          # int -> str
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.int32)

        self.term_ids: Dict[str, int] = {}        # stem -> term ordinal
        self.post_off: np.ndarray = np.zeros(1, dtype=np.int64)
        self.post_doc: np.ndarray = np.zeros(0, dtype=np.int32)
        self.post_tf: np.ndarray = np.zeros(0, dtype=np.int32)

        self.N: int = 0
        self.avg_doc_len: float = 0.0
        # Pre-computed doc_len[i] / avg_doc_len for BM25 speedup.
        self.doc_len_ratio: np.ndarray = np.zeros(0, dtype=np.float32)

    def build(self, corpus: Iterable[Tuple[str, str]]) -> None:
        """corpus: iterable of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.iter_corpus().
        """
        self.doc_id_map = []
        raw_lens: List[int] = []

        # Everything that can be decided per *type* rather than per
        # *occurrence* is deferred out of this loop. Stemming was already
        # deferred; stopword and minimum-length filtering now are too. The
        # streaming pass therefore does no Python-level work per token at
        # all beyond one dict lookup: it records unstemmed, unfiltered
        # vocabulary ordinals, and the ~207k-entry remap below applies the
        # stoplist, the length rule and the stemmer exactly once each.
        # Filtering here instead cost 11.8s of listcomp over 29M tokens.
        #
        # The C++ extension does the same thing without materialising the
        # 29M intermediate Python objects. It cannot express the compound
        # tokenizer, so that option stays on the Python path.
        if _fast_tokenizer is not None and not _COMPOUNDS:
            raw_terms, all_raw = _stream_fast(corpus, self.doc_id_map, raw_lens)
        else:
            raw_terms, all_raw = _stream_python(corpus, self.doc_id_map, raw_lens)

        self.N = len(self.doc_id_map)

        # Apply the stoplist, the length rule and the stemmer once per
        # distinct raw type, then collapse types that share a stem into a
        # single term ordinal. Types the tokenizer drops map to -1 and are
        # masked out of the posting stream below.
        min_len = _MIN_TOKEN_LEN
        stop = _STOPWORDS
        vocab: Dict[str, int] = {}
        remap = np.empty(len(raw_terms), dtype=np.int64)
        for raw_ordinal, raw_term in enumerate(raw_terms):
            if len(raw_term) < min_len:
                remap[raw_ordinal] = -1
                continue
            token = raw_term.decode("ascii")
            if token in stop:
                remap[raw_ordinal] = -1
                continue
            stem = _stem(token)
            term_id = vocab.get(stem)
            if term_id is None:
                term_id = len(vocab)
                vocab[stem] = term_id
            remap[raw_ordinal] = term_id

        self.term_ids = vocab
        num_terms = len(vocab)

        if all_raw.size == 0:
            self.doc_len = np.zeros(self.N, dtype=np.int32)
            self.avg_doc_len = 0.0
            self.doc_len_ratio = self._compute_doc_len_ratio()
            self.post_off = np.zeros(num_terms + 1, dtype=np.int64)
            self.post_doc = np.zeros(0, dtype=np.int32)
            self.post_tf = np.zeros(0, dtype=np.int32)
            return

        # Inversion, written to keep peak memory down rather than to be
        # pretty: the grading machine has 8 GB and the evaluation collection
        # is larger than the released one, so every full-length temporary
        # here is one that could push the build into swap or an OOM. Each
        # intermediate is therefore freed as soon as it is consumed, and the
        # combined key is built in place.
        #
        # Sort by (term, doc) via a combined key, then collapse repeated
        # (term, doc) pairs into one posting whose tf is the run length.
        # An int64 key is safe: term < num_terms, doc < N, and
        # num_terms * N is far inside int64 range at this scale.
        key = remap[all_raw]                   # int64, doubles as the sort key
        del all_raw

        doc_of_token = np.repeat(np.arange(self.N, dtype=np.int32),
                                 np.asarray(raw_lens, dtype=np.int64))

        # Drop the occurrences of stopped / too-short types. Because the
        # filter now runs here rather than during tokenization, doc_len is
        # whatever survives it — count it back off the surviving stream
        # rather than trusting the raw token counts.
        kept = key >= 0
        if not kept.all():
            key = key[kept]
            doc_of_token = doc_of_token[kept]
        del kept

        self.doc_len = np.bincount(doc_of_token,
                                   minlength=self.N).astype(np.int32)
        self.avg_doc_len = key.size / self.N if self.N > 0 else 0.0
        self.doc_len_ratio = self._compute_doc_len_ratio()

        key *= np.int64(self.N)
        np.add(key, doc_of_token, out=key)     # in place; no int64 temporary
        del doc_of_token

        # Default (quick)sort, not "stable": we only need the sorted values,
        # and equal int64 keys are indistinguishable, so stability buys
        # nothing while mergesort would allocate a full-length scratch buffer.
        key.sort()

        # np.unique(..., return_counts=True) would copy the whole key array
        # again internally. The array is already sorted, so find run
        # boundaries directly instead.
        boundaries = np.flatnonzero(key[1:] != key[:-1])
        starts = np.empty(boundaries.size + 2, dtype=np.int64)
        starts[0] = 0
        np.add(boundaries, 1, out=starts[1:-1])
        starts[-1] = key.size
        del boundaries

        uniq_key = key[starts[:-1]]
        self.post_tf = np.diff(starts).astype(np.int32)
        del key, starts

        post_term = uniq_key // self.N
        self.post_doc = (uniq_key % self.N).astype(np.int32)
        del uniq_key

        counts_per_term = np.bincount(post_term, minlength=num_terms)
        self.post_off = np.zeros(num_terms + 1, dtype=np.int64)
        np.cumsum(counts_per_term, out=self.post_off[1:])

        self._relabel_terms_in_sorted_order()

    def _relabel_terms_in_sorted_order(self) -> None:
        """Renumber term ordinals so they follow lexicographic order.

        Ordinals are assigned in first-seen order during the streaming
        pass, which is arbitrary. Sorting them lets save() write the term
        dictionary front-coded (adjacent terms share long prefixes) without
        also having to store a permutation back to the original ordinals —
        the permutation costs more than front-coding saves.
        """
        terms = list(self.term_ids)
        old_ids = np.fromiter((self.term_ids[t] for t in terms),
                              dtype=np.int64, count=len(terms))
        order = np.argsort(np.array(terms, dtype=object), kind="stable")

        sorted_terms = [terms[i] for i in order]
        sorted_old_ids = old_ids[order]
        self.term_ids = {t: i for i, t in enumerate(sorted_terms)}

        counts = (self.post_off[1:] - self.post_off[:-1])[sorted_old_ids]
        new_off = np.zeros(len(sorted_terms) + 1, dtype=np.int64)
        np.cumsum(counts, out=new_off[1:])

        # Gather each term's posting block into its new position.
        old_starts = self.post_off[:-1][sorted_old_ids]
        gather = np.repeat(old_starts - new_off[:-1], counts) + \
            np.arange(self.post_doc.size, dtype=np.int64)
        self.post_doc = self.post_doc[gather]
        self.post_tf = self.post_tf[gather]
        self.post_off = new_off

    def _compute_doc_len_ratio(self) -> np.ndarray:
        if self.avg_doc_len > 0:
            return (self.doc_len.astype(np.float32) / np.float32(self.avg_doc_len))
        return np.zeros(self.N, dtype=np.float32)

    def postings_for(self, term: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (docs, tfs) numpy array slices (views, not copies) for
        `term`, or two empty arrays if the term isn't in the vocabulary."""
        tid = self.term_ids.get(term)
        if tid is None:
            empty = np.zeros(0, dtype=np.int32)
            return empty, empty
        lo, hi = self.post_off[tid], self.post_off[tid + 1]
        return self.post_doc[lo:hi], self.post_tf[lo:hi]

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        tid = self.term_ids.get(term)
        if tid is None:
            return 0
        return int(self.post_off[tid + 1] - self.post_off[tid])

    def save(self, index_dir: str) -> None:
        """Persist everything scorers need to `index_dir`, so `load()` can
        reconstruct this object in a fresh process with no memory of
        `build()` ever having run. Called from retrieve.build_index().

        The file is a small JSON header followed by independently
        zlib-compressed binary blocks. Postings are stored as per-term
        doc-id gaps and term frequencies in the byte-bucketed encoding from
        submission/codec.py, and the term dictionary is front-coded; see
        that module for why that beats variable-byte here. Nothing that can
        be recomputed at load time is written: absolute doc ids, document
        length ratios and IDF values are all derived in load().
        """
        os.makedirs(index_dir, exist_ok=True)
        path = os.path.join(index_dir, "index.bin")

        terms_sorted = sorted(self.term_ids, key=self.term_ids.get)
        gaps = codec.encode_doc_gaps(self.post_doc, self.post_off)
        gap1, gap2, gap3 = codec.encode_bucketed(gaps)
        tf1, tf2, tf3 = codec.encode_bucketed(self.post_tf.astype(np.uint32))
        len1, len2, len3 = codec.encode_bucketed(self.doc_len.astype(np.uint32))
        ids_payload, idlen1, idlen2, idlen3 = codec.encode_strings(self.doc_id_map)
        counts = (self.post_off[1:] - self.post_off[:-1]).astype(np.uint32)
        cnt1, cnt2, cnt3 = codec.encode_bucketed(counts)

        blocks = {
            "terms": codec.front_code(terms_sorted),
            "counts1": cnt1, "counts2": cnt2, "counts3": cnt3,
            "gaps1": gap1, "gaps2": gap2, "gaps3": gap3,
            "tf1": tf1, "tf2": tf2, "tf3": tf3,
            "doclen1": len1, "doclen2": len2, "doclen3": len3,
            "docids": ids_payload,
            "docidlen1": idlen1, "docidlen2": idlen2, "docidlen3": idlen3,
        }

        header = {
            "format": _INDEX_FORMAT,
            "N": self.N,
            "num_terms": len(terms_sorted),
            "num_postings": int(self.post_doc.size),
            "avg_doc_len": self.avg_doc_len,
            # Persisted so load() can restore the exact tokenizer this index
            # was built with — querying an index with a different tokenizer
            # degrades silently rather than failing loudly.
            "tokenizer": tokenizer_config(),
            "block_order": list(blocks),
            "block_sizes": {},
        }

        # Level 6 rather than 9: on these streams level 9 buys well under
        # 1% and costs several seconds of index build time, which is itself
        # a scored metric.
        #
        # The blocks are compressed on a thread pool because zlib releases
        # the GIL around deflate, so this is one of the few places in a
        # pure-Python build where threads give real parallelism. The blocks
        # are independent by construction (each is decompressed on its own
        # in load()), and results are collected by index rather than by
        # completion order, so the output file is byte-for-byte identical
        # to the serial version regardless of how the threads interleave.
        names = list(blocks)
        raws = [blocks[name] for name in names]
        max_workers = min(len(raws), (os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            payloads = list(pool.map(lambda raw: zlib.compress(raw, level=6), raws))
        for name, packed in zip(names, payloads):
            header["block_sizes"][name] = len(packed)

        header_bytes = json.dumps(header).encode("utf-8")
        with open(path, "wb") as f:
            f.write(_MAGIC)
            f.write(len(header_bytes).to_bytes(4, "little"))
            f.write(header_bytes)
            for packed in payloads:
                f.write(packed)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an InvertedIndex purely from what save() wrote to
        `index_dir`. Called in a fresh process — do not rely on any state
        other than what's actually on disk in `index_dir`.
        """
        path = os.path.join(index_dir, "index.bin")
        with open(path, "rb") as f:
            blob = f.read()

        if blob[:len(_MAGIC)] != _MAGIC:
            raise ValueError("corrupt index: bad magic bytes")
        pos = len(_MAGIC)
        header_len = int.from_bytes(blob[pos:pos + 4], "little")
        pos += 4
        header = json.loads(blob[pos:pos + header_len].decode("utf-8"))
        pos += header_len
        if header.get("format") != _INDEX_FORMAT:
            raise ValueError(
                f"index format {header.get('format')!r} does not match "
                f"this build ({_INDEX_FORMAT!r}); rebuild the index")

        blocks = {}
        for name in header["block_order"]:
            size = header["block_sizes"][name]
            blocks[name] = zlib.decompress(blob[pos:pos + size])
            pos += size

        # Restore the tokenizer this index was built with before anything
        # queries it.
        tok = header.get("tokenizer")
        if tok:
            configure_tokenizer(**tok)

        index = cls()
        index.N = header["N"]
        index.avg_doc_len = header["avg_doc_len"]

        terms = codec.front_decode(blocks["terms"], header["num_terms"])
        index.term_ids = {term: i for i, term in enumerate(terms)}

        counts = codec.decode_bucketed(
            blocks["counts1"], blocks["counts2"], blocks["counts3"])
        index.post_off = np.zeros(len(terms) + 1, dtype=np.int64)
        np.cumsum(counts.astype(np.int64), out=index.post_off[1:])

        gaps = codec.decode_bucketed(
            blocks["gaps1"], blocks["gaps2"], blocks["gaps3"])
        index.post_doc = codec.decode_doc_gaps(gaps, index.post_off)
        index.post_tf = codec.decode_bucketed(
            blocks["tf1"], blocks["tf2"], blocks["tf3"]).astype(np.int32)
        index.doc_len = codec.decode_bucketed(
            blocks["doclen1"], blocks["doclen2"], blocks["doclen3"]).astype(np.int32)
        index.doc_id_map = codec.decode_strings(
            blocks["docids"], blocks["docidlen1"],
            blocks["docidlen2"], blocks["docidlen3"])

        index.doc_len_ratio = index._compute_doc_len_ratio()
        return index
