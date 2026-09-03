"""
submission/stoplists.py — stopword lists available to the tokenizer.

Inlined as literals rather than pulled from `nltk.corpus.stopwords`
on purpose: the grading harness builds the index inside a container run
with `--network none`, and `nltk.corpus.stopwords` needs a downloaded
data package that is not part of the base image. A literal list also
makes the submission reproducible and self-describing for the report.

Two lists:

* `MINIMAL` — the 50-word hand-rolled list the first submission used.
* `STANDARD` — a conventional ~420-word English stoplist of the
  Indri/Lemur lineage. Much more aggressive; removes most closed-class
  vocabulary. Removing the high-document-frequency head also shrinks the
  postings file substantially, so this list is simultaneously a retrieval
  quality knob and an index-size knob.

Which one ships is decided by cross-validated nDCG@10 in
`scripts/tune.py`, not by assumption.
"""

MINIMAL = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "has", "have", "how",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "there", "these", "this", "to", "was", "were", "what", "when", "which", "who",
    "will", "with", "would", "any", "also", "among", "about", "than", "then",
})

STANDARD = frozenset("""
a about above according across after afterwards again against albeit all almost
alone along already also although always am among amongst an and another any
anybody anyhow anyone anything anyway anywhere apart are around as at av be
became because become becomes becoming been before beforehand behind being
below beside besides between beyond both but by can cannot canst certain cf
choose contrariwise cos could cu day do does doesn doing dost doth double down
dual during each either else elsewhere enough et etc even ever every everybody
everyone everything everywhere except excepted excepting exception exclude
excluding exclusive far farther farthest few ff first for formerly forth
forward from front further furthermore furthest get go had halves hardly has
hast hath have he hence henceforth her here hereabouts hereafter hereby herein
hereto hereupon hers herself him himself hindmost his hither hitherto how
however howsoever i if in inasmuch inc include included including indeed
indoors inside insomuch instead into inward inwards is it its itself just kind
kg km last latter latterly less lest let like little ltd many may maybe me
meantime meanwhile might moreover most mostly more mr mrs ms much must my
myself namely need neither never nevertheless next no nobody none nonetheless
noone nor not nothing notwithstanding now nowadays nowhere of off often ok on
once one only onto or other others otherwise ought our ours ourselves out
outside over own per perhaps plenty provide quite rather really round said
sake same sang save saw see seeing seem seemed seeming seems seen seldom
selves sent several shalt she should shown sideways since slept slew slung
slunk smote so some somebody somehow someone something sometime sometimes
somewhat somewhere spake spat spoke spoken sprang sprung stave staves still
such supposing than that the thee their them themselves then thence
thenceforth there thereabout thereabouts thereafter thereby therefore therein
thereof thereon thereto thereupon these they this those thou though thrice
through throughout thru thus thy thyself till to together too toward towards
ugh unable under underneath unless unlike until up upon upward upwards us use
used using very via vs want was we week well were what whatever whatsoever
when whence whenever whensoever where whereabouts whereafter whereas whereat
whereby wherefore wherefrom wherein whereinto whereof whereon wheresoever
whereto whereunto whereupon wherever wherewith whether whew which whichever
whichsoever while whilst whither who whoa whoever whole whom whomever
whomsoever whose whosoever why will wilt with within without worse worst would
wow ye yet year yippee you your yours yourself yourselves
""".split())

BY_NAME = {
    "minimal": MINIMAL,
    "standard": STANDARD,
    "none": frozenset(),
}
