# Phase 2: baselines and the teacher

Which English Wikipedia edits get reverted, and how well can different models
tell in advance? This report compares simple rules, gradient-boosted trees,
Wikimedia's own revert-risk models and a fine-tuned transformer (the teacher
that phase 3 will distil into a small browser model). All of them are scored on
the same three test days, 3 to 5 November 2026, which come after everything the
models were trained on.

**Short version.**

- **The teacher is best on every measure.** It is DeBERTa-v3-base reading the
  diff, the edit summary and the editor's details as text. Its PR-AUC is 0.59,
  and 85% of the edits it ranks in the top 2% were reverted.
- **The other models:** the best gradient-boosted tree reaches 0.50 and 70%,
  and Wikimedia's recommended setup 0.44 and 68%.
- **The lead holds on each test day.** The teacher is 0.09 ahead of the tree on
  each of the three days separately.
- **Where it's strongest:** anonymous edits and new accounts.
- **Where nothing works:** edits by registered editors with 100 or more edits
  are three-quarters of the stream and 18% of reverts. Every model is barely
  better than chance on them.

An earlier run with a single, partial test day (29 October) is summarised at
the end. The three-day results are close to it, and a little better because
the training set is two and a half times larger.

## Data

- **Source:** the phase 1 collector, 25 October to 5 November 2026 (1.40
  million article edits).
- **Scope:** edits without the bot flag (963,000). Bot edits were 31% of the
  stream, and only 0.19% of them were reverted.
- **Label:** reverted by someone else within 48 hours. I used MediaWiki's
  `mw-reverted` tag, then the page history to find who reverted and when.
  Self-reverts count as not reverted. Overall, 5.5% of edits are reverted.
- **Dropped:**
  - edits with no label (1,434: the page or revision has been deleted since);
  - edits whose diff couldn't be fetched (1,379, see "Leakage" below);
  - edits still waiting for their diff when the data was exported (52,443,
    almost all on 29 and 30 October). The collector first fetched diffs one at a
    time and fell behind, and it works through the backlog newest first. So the
    gap is by time of arrival, not by what kind of edit it was: on those two
    days, 5.7% of the waiting edits were reverted, against 5.5% of the fetched
    ones.

### Split

| Set | Days | Edits | Reverted |
|---|---|---|---|
| Train | 25 October-1 November | 408,921 | 20,599 (5.0%) |
| Validation | 2 November | 60,021 | 2,899 (4.8%) |
| Test | 3-5 November, whole days | 247,471 | 14,068 (5.7%) |

No page appears in more than one set. When a page shows up in a later set, its
edits are removed from the earlier one, so the test days stay an untouched
slice of live traffic. Busy pages get edited day after day, so this removed 29%
of train edits and 31% of validation edits.

## Models

**Rules.** None of these is trained. Each score level is mapped to the revert
rate seen at that level on the validation day.

- An anonymous editor removing 500 bytes or more.
- A word from a short profanity list being added.
- A points score combining those two with:
  - blanking most of a page;
  - shouting in capitals;
  - no edit summary from an account with under 10 edits.

**Gradient-boosted trees** (LightGBM, early stopping on the validation day):

- **Metadata only:**
  - the editor: anonymous or not, edit count, account age, user groups;
  - the edit: sizes, edit summary length, change tags, time of day.
- **Metadata plus cheap statistics of the diff text:**
  - characters added and removed, and the share of capitals;
  - links, references, templates, URLs and repeated characters;
  - profanity.

**Teacher:** `microsoft/deberta-v3-base` fine-tuned on each edit written out as
text, like this:

```
anonymous editor (temporary account), account under a week, 5 edits
edit: size 72835 to 72855 (+20), 0 lines added, 0 removed, 1 changed
tags: mobile edit, mobile web edit
page: Student List
summary: /* */ link [[Plenary session|plenums]]
removed: (nothing)
added: [[Plenary session| ]]
```

- **Training:** one epoch over 408,921 edits, batch 32, learning rate 2e-5,
  bf16, inputs cut at 256 tokens. It took 73 minutes on an RTX 3080. In the
  first run, a second epoch made the validation score slightly worse, so I
  stopped at one.
- **Class imbalance:** I trained on the natural 5.0% positive rate with plain
  cross-entropy, starting the output bias at that rate, and chose checkpoints
  by PR-AUC. Reweighting the classes would have distorted the probabilities
  that calibration then has to fix.
- **Calibration:** two-parameter Platt scaling fitted on the validation day
  (a = 0.80, b = -0.32). This lowered the test Brier score from 0.0347 to
  0.0342. A single temperature did almost as well this time (0.0343).

**Wikimedia's revert-risk models** on Lift Wing, scored through the public
API:

- **Language-agnostic:** gradient-boosted trees over article-quality features
  of the two revisions, plus account age, edit count and number of user groups.
  Trained on 2022 edits.
- **Multilingual:** a language model that reads the change. Wikimedia's model
  card recommends it for anonymous edits and the language-agnostic model for
  everything else.

Scoring all 247,471 test edits through the public API would have taken about
a day. Instead, I scored a random 25,000 of them, plus the 3,551 anonymous edits
among those with the multilingual model. The API couldn't score 19 of the
sample edits (0.1%), whose revisions have since been deleted or hidden.

## Results

### All test edits

Every model except Wikimedia's was scored on all 247,471 test edits. Square
brackets are 95% bootstrap intervals over edits (1,000 resamples).

| Model | PR-AUC | Precision at top 2% | Recall at top 2% | Recall at top 5% | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Teacher (DeBERTa-v3-base, Platt-calibrated) | 0.589 [0.580-0.597] | 0.85 [0.84-0.86] | 0.30 | 0.51 | 0.917 | 0.0342 |
| GBDT, metadata + diff statistics | 0.496 [0.487-0.505] | 0.70 [0.69-0.71] | 0.25 | 0.46 | 0.909 | 0.0381 |
| GBDT, metadata only | 0.454 [0.445-0.464] | 0.64 [0.62-0.65] | 0.22 | 0.43 | 0.904 | 0.0396 |
| Rules, points score | 0.240 [0.233-0.246] | 0.40 [0.39-0.41] | 0.14 | 0.27 | 0.776 | 0.0467 |
| Rule: anonymous + removal of 500+ bytes | 0.090 [0.086-0.094] | 0.17 [0.16-0.18] | 0.06 | 0.09 | 0.512 | 0.0528 |
| Rule: profanity added | 0.074 [0.072-0.077] | 0.18 [0.17-0.19] | 0.06 | 0.10 | 0.518 | 0.0536 |

Base rate: 5.7% of test edits were reverted. Flagging edits at random would give
PR-AUC and precision of about 0.057.

**Teacher against the best tree.**

- **PR-AUC:** the teacher is 0.093 higher (paired bootstrap interval 0.087 to
  0.099).
- **Precision at the top 2%**, the alert rate the demo will use: 85% of the
  teacher's flags were reverted, against 70% of the tree's.
- **Recall:** at that rate, the teacher's flags catch 30% of all reverted
  edits. Most reverts aren't in the top 2% of any model.

### Day by day

The bootstrap intervals above only capture variation between edits. Splitting
the test set by day shows how much the numbers move from one day to the next.

| Test day | Edits | Reverted | Teacher PR-AUC | Teacher precision at top 2% | Tree PR-AUC | Tree precision at top 2% |
|---|---|---|---|---|---|---|
| 3 November | 82,978 | 5.5% | 0.586 | 0.84 | 0.496 | 0.70 |
| 4 November | 80,734 | 5.8% | 0.616 | 0.89 | 0.515 | 0.72 |
| 5 November | 83,759 | 5.7% | 0.566 | 0.82 | 0.479 | 0.69 |

PR-AUC moves by about 0.05 between days, which is several times the bootstrap
interval. The gap between the two models stays at 0.09-0.10 on every day.

### Against Wikimedia's models

All models scored on the same random 25,000 test edits (24,981 after the
edits the API couldn't score).

| Model | PR-AUC | Precision at top 2% | Recall at top 2% | Recall at top 5% | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Teacher | 0.589 [0.565-0.613] | 0.85 [0.81-0.89] | 0.29 | 0.50 | 0.917 | 0.0352 |
| GBDT, metadata + diff statistics | 0.494 [0.467-0.520] | 0.70 [0.64-0.74] | 0.24 | 0.45 | 0.908 | 0.0391 |
| GBDT, metadata only | 0.447 [0.419-0.473] | 0.63 [0.58-0.67] | 0.22 | 0.42 | 0.903 | 0.0407 |
| Wikimedia revert-risk, recommended setup | 0.445 [0.417-0.474] | 0.68 [0.64-0.73] | 0.24 | 0.40 | 0.873 | 0.1835 |
| Wikimedia revert-risk, language-agnostic | 0.306 [0.284-0.329] | 0.41 [0.37-0.46] | 0.14 | 0.30 | 0.863 | 0.2020 |
| Rules, points score | 0.250 [0.232-0.271] | 0.40 [0.36-0.45] | 0.14 | 0.28 | 0.781 | 0.0473 |

![Precision-recall curves on the 25,000-edit sample](phase2/pr_curves.png)

- **Recommended setup:** the multilingual model for anonymous edits and the
  language-agnostic one for the rest.
  - It ranks level with the tree on metadata alone (PR-AUC 0.445 against
    0.447), and its precision at the top 2% is a little higher (0.68 against
    0.63).
  - The teacher is 0.144 ahead of it (paired interval 0.122 to 0.167).
- **Language-agnostic model alone:** clearly weaker, at 0.306.
- **Anonymous edits** (3,547 in the sample, 24.9% reverted):

  | Model | PR-AUC on anonymous edits |
  |---|---|
  | Teacher | 0.703 [0.675-0.727] |
  | Wikimedia multilingual | 0.602 [0.570-0.635] |
  | GBDT, metadata + diff statistics | 0.572 [0.538-0.607] |
  | GBDT, metadata only | 0.519 |
  | Wikimedia language-agnostic | 0.373 |

  - Wikimedia's multilingual model is ahead of my best tree by 0.030, but the
    interval (-0.003 to 0.062) includes zero. In the first run it was clearly
    ahead.
  - The teacher is 0.101 ahead of the multilingual model (interval 0.073 to
    0.128).
- **Registered editors:** the language-agnostic model is weakest here. With 2%
  of edits flagged, the recommended setup catches 5.6% of the reverts by
  registered editors, against 18% for the teacher.
- **Fairness of the comparison:** it isn't a level comparison.
  - Wikimedia's models were trained on 2022 edits with a different label (any
    revert), and they serve every language.
  - Mine were trained on this exact label, from the eight days just before.
  - On the other hand, Wikimedia's editor features are computed when the model
    is called, days after the edit, which helps them (see "Leakage").

### Calibration

![Calibration on the 25,000-edit sample](phase2/reliability.png)

- **Teacher:** after Platt scaling, its probabilities track the observed revert
  rate closely up to about 0.6. Its Brier score (0.0342 on all test edits) is
  the best of all models.
- **Teacher, two weak spots:**
  - Edits it scores around 0.15 (the 90th to 95th percentile) were reverted
    21% of the time, so it is slightly under-confident there.
  - Edits around 0.83 (the 98th to 99th percentile) were reverted 73% of the
    time, so it is over-confident there.

  Over all test edits, those scoring 0.7 to 0.8 were reverted 66% of the time.
- **Tree:** well calibrated all the way up. It is slightly under-confident
  around 0.7.
- **Wikimedia's models:** their raw outputs aren't calibrated to this revert
  rate.
  - The median unreverted edit gets 0.33 from the language-agnostic model, and
    the median reverted edit 0.85, which is why its Brier score is poor.
  - That doesn't affect their ranking metrics.

## Where the models fall short

These figures cover all 247,471 test edits. "Share of flags" is the share of
each model's top-2% flags that fall in the slice, and "recall" is the share of
the slice's reverts that those flags catch.

| Slice | Edits | Reverted | Share of all reverts | Teacher PR-AUC | Tree PR-AUC | Teacher: share of flags | Teacher: precision of flags | Teacher: recall |
|---|---|---|---|---|---|---|---|---|
| Anonymous (all on temporary accounts) | 35,036 | 24.2% | 60% | 0.700 | 0.577 | 75% | 0.86 | 0.38 |
| Registered, under 100 edits | 23,522 | 13.1% | 22% | 0.617 | 0.533 | 24% | 0.84 | 0.32 |
| Registered, 100+ edits | 188,913 | 1.3% | 18% | 0.087 | 0.082 | 0.8% | 0.87 | 0.01 |
| Input cut at 256 tokens | 34,703 | 6.3% | 16% | 0.557 | 0.523 | 15% | 0.82 | 0.28 |
| Fits in 256 tokens | 212,768 | 5.6% | 84% | 0.595 | 0.490 | 85% | 0.86 | 0.30 |

- **Experienced editors.**
  - Edits by registered editors with 100+ edits are 76% of the stream and 18%
    of the reverts.
  - No model does much better than the 1.3% base rate on them: the teacher's
    PR-AUC is 0.087 and the tree's 0.082.
  - The teacher's flags almost never land on them (0.8% of flags), and they
    catch 1% of these reverts.
  - These reverts are mostly content disputes, style changes and good-faith
    mistakes, which the diff alone rarely gives away. The demo should say
    plainly that it misses these.
- **Who gets flagged.**
  - 75% of the teacher's flags are anonymous edits, which are 14% of all edits
    but 60% of reverts.
  - Precision on its flags is the same for anonymous editors (86%) and
    registered ones (84-87%). So it is not flagging anonymous edits more
    readily than their revert rate supports.
  - The demo should still avoid framing a flag as a judgement on the editor.
- **Long edits.**
  - 14% of test edits don't fit in 256 tokens.
  - On those, the teacher's lead over the tree shrinks from 0.105 to 0.034,
    and its recall at the top 2% (0.28) falls below the tree's (0.30).
  - A run with a 384-token limit was started to test this, but it was cut off
    when the PC shut down (see "Next").
- **Diff text loses context.** For a changed line, the input shows only the
  inserted and removed words, so turning a word into a link reads as
  `added: [[ ]]`. Showing a few words either side should help.
- **Labels.** "Reverted" isn't "vandalism".
  - Good-faith edits get reverted too.
  - Some bad edits are fixed by hand instead of reverted, so they count as not
    reverted.
  - The model learns what gets reverted, so the demo will say "likely to be
    reverted", not "vandalism".

## Speed

| Model | Time per edit, one at a time |
|---|---|
| Rules | under 0.1 ms |
| Gradient-boosted tree | 1.2-1.3 ms (2-core Xeon, mostly Python overhead) |
| Teacher on the RTX 3080 | 20.0 ms (90th percentile 21.8 ms); 12.6 ms in the first run |
| Teacher on the desktop CPU | 69 ms (90th percentile 113 ms) |
| Wikimedia, language-agnostic, through the public API from Australia | about 200 ms round trip |
| Wikimedia, multilingual, same | 300-1,000 ms round trip |

- **Teacher timings:** these include tokenisation. The GPU figure varies
  between runs on the same card by more than the model does, so treat it as
  rough.
- **Phase 3 budget:** under 50 ms per edit in a browser on a mid-range laptop,
  and under 15 MB to download.
- **The gap:** the teacher alone is 738 MB in float32 and 69 ms on a desktop
  CPU, so distillation is needed for both the size and the speed.

## Leakage risks and what I did about them

- **Pages in more than one set:** removed, as described in "Split".
- **Same editors in train and test:** many test edits are by editors who also
  appear in training (68% in the first run).
  - The models never see a username, only the edit count, account age and
    groups, so they can't memorise individuals.
  - I left this as it is, because that's what the live demo will face too.
- **Details fetched after the edit:**
  - **Editor details and change tags:** read about 80 seconds after each edit
    (median), so a fast editor's count includes a few later edits. The live
    demo will query them a few seconds after the edit, which is close enough.
  - **The `mw-reverted` tag:** this is the label, so it is removed from the
    save-time tags.
  - **Tags read 48 hours later:** never used as features.
- **Diffs fetched late:** diffs are fixed once an edit is saved, so fetching
  them days later gives the same text.
  - But if a revision was deleted or hidden before its diff was fetched, the
    fetch fails, and that happens more often to bad edits: 29% of failed
    fetches were reverted, against 5.5% overall.
  - Keeping those edits without a diff would let a model learn "no diff means
    bad edit", so they are dropped. There were 1,379, 0.14% of the total.
- **Edits still waiting for a diff:** dropped. They were reverted at the same
  rate as the rest (see "Data"), so dropping them doesn't tilt the label.
- **Wikimedia's models had an edge, not a handicap.** Lift Wing computes its
  editor features when it's called, three to six days after the edits here, so
  its edit counts include later edits. It was also trained on a different label
  (any revert, from 2022). Both caveats apply when comparing.

## The first run (29 October test day)

The first version of this report used three days of training (161,361 edits),
28 October for validation, and the first 16 hours of 29 October as the test
set (51,351 edits). It was limited by the diff backlog at the time.

| Model | First run: PR-AUC | First run: precision at top 2% | This run: PR-AUC | This run: precision at top 2% |
|---|---|---|---|---|
| Teacher | 0.532 | 0.78 | 0.589 | 0.85 |
| GBDT, metadata + diff statistics | 0.466 | 0.67 | 0.496 | 0.70 |
| GBDT, metadata only | 0.436 | 0.62 | 0.454 | 0.64 |
| Wikimedia, recommended setup | 0.430 | 0.64 | 0.445 | 0.68 |
| Rules, points score | 0.240 | 0.38 | 0.240 | 0.40 |

- **The ordering is the same.**
- **The teacher gained most** from the larger training set (+0.057), the trees
  less (+0.02-0.03), and the rules nothing.
- **Comparability:** the test days differ, and the day-by-day table above shows
  PR-AUC moving by 0.05 between days. So only the gaps between models compare
  directly across the two runs, not the absolute numbers.
- **Files:** the first run's charts and figures are in `reports/phase2/v1/`.

## Next

- **The 384-token teacher:** rerun the run that was cut off, to see whether a
  longer input closes the gap on long edits. If it does, the student models in
  phase 3 should use the longer input too, if the browser budget allows.
- **Phase 3:** distil the teacher into two student sizes, export them to ONNX,
  quantise to int8, and measure them in the browser against the budget.

## Reproducing

```
python -m training.train_teacher --split v2 --epochs 1 --score-train --max-len 256 --exports EXPORTS --out runs/teacher-v2-256   # GPU
python -m training.baselines     --split v2 --exports EXPORTS --out runs/baselines-v2
python -m training.liftwing      --rev-ids v2_test_sample.parquet --out runs/v2-liftwing.jsonl
python -m training.liftwing      --rev-ids v2_test_sample_anon.parquet --out runs/v2-liftwing-ml.jsonl \
                                 --model revertrisk-multilingual
python -m training.compare       --split v2 --exports EXPORTS --teacher runs/teacher-v2-256 --baselines runs/baselines-v2 \
                                 --liftwing runs/v2-liftwing.jsonl --liftwing-ml runs/v2-liftwing-ml.jsonl \
                                 --tokenizer microsoft/deberta-v3-base --max-len 256 \
                                 --title "Precision and recall, 3 to 5 November 2026" --out reports/phase2
```

The sample files are a random draw of 25,000 `rev_id`s from the test set, and
the anonymous edits within that draw.
