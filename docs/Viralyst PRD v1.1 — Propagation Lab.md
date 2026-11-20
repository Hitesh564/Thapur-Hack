# Viralyst PRD v1.1 — Propagation Lab

Nov 20, 2026 · @Hitesh Jindal

## Summary

Viralyst is a pre-publication lab for short-form video. It scores a draft with a model trained on real engagement data, simulates how a synthetic audience watches and shares it through a modelled network, and lets the creator run controlled experiments on that simulation, with an honest statement of how far each result can be trusted.

**The claim we pitch:** "From viral scores to propagation experiments you can trust. Every verdict comes with its seed-robustness, its assumption-robustness and an evidence label."

### What changed from PRD v1.0

- **The model target** is the continuous SnapUGC label, trained on a small chunk, with SROCC, PLCC and baselines instead of three classes and accuracy.
- **Round-0 attention is anchored** to the model's predicted score, so the first step of every cascade is data-grounded.
- **The main hook experiment uses a real alternate opening clip,** re-scored by the trained model on real frames.
- **Relevance comes from a joint embedding** of persona interests and video frames, not from LLM-labelled topics.
- **The audience is 3,000 to 5,000 viewers** sampled from LLM-written archetypes, so hubs, bridges and depth show up.
- **Two propagation channels** (recommendation and social shares) are accounted for separately.
- **Four Lab signatures are added:** reproduction-number (R) diagnosis, reach attribution, a tipping-point curve and two-level robustness over seeds and assumptions.
- **An Evidence ledger** tags every number as Learned, Calibrated, Assumed or Simulated.

### What is trained and what is assumed

| Component | Status |
| --- | --- |
| Video to engagement-percentile prior | **Trained** on a small SnapUGC chunk |
| Round-0 stay probability | **Calibrated** to the prior; the base rate is an assumption |
| Persona-video relevance | **Computed** from pretrained embeddings, not trained by us |
| Persona tendencies, graph, distribution policy | **Assumed**, with stated ranges that the robustness loop samples |
| Everything an experiment outputs | **Simulated** |

This table is also a screen in the product (the Evidence ledger), and it is the answer to the first question any mentor will ask.

## Scope

Viralyst helps a creator test a draft video before posting. It never promises views, and it never claims to be a platform's algorithm.

### Users and the core question

- **Primary:** short-form creators with a draft ready to post.
- **Secondary:** small brand teams, student creators, content strategists, and the hackathon judges.

The question we answer: "I have this video ready. Which audiences might respond, where could it stall, and what change gives the biggest, most reliable improvement?"

### Input

Any category of short-form video (comedy, food, fitness, education, product and so on), MP4 preferred, 5 to 60 seconds, vertical preferred, audio optional. The product is content-agnostic and is not an AI-and-tech tool.

### Goals

1. Analyse the video and its opening hook automatically.
2. Provide a **trained** engagement prior with held-out evidence, using a small SnapUGC chunk.
3. Simulate a heterogeneous audience efficiently, with no LLM call per viewer.
4. Make propagation observable and explainable.
5. Run matched experiments and report how stable each verdict is.
6. Show what is evidence and what is assumption on every number.

### Non-goals

- Guaranteeing virality or predicting exact views.
- Replicating Instagram, TikTok, Snapchat or YouTube ranking.
- Claiming personas are real people or represent real demographics.
- Claiming causal effects of a hook change.
- One LLM call per persona per round.
- Account access, scraping creator analytics, or building a video editor.
- Training a video foundation model from scratch.

### Claims

| We say | We never say |
| --- | --- |
| "We simulate how a short video may resonate across a synthetic audience." | "We predict how many views you will get." |
| "We compare scenarios under controlled assumptions and report how stable the verdict is." | "Changing your hook will increase reach by X%." |
| "The trained model is one signal; behaviour and propagation are simulated." | "Our model predicts each persona's behaviour." |
| "This is our own documented distribution policy." | "We recreated Instagram's algorithm." |

## USPs and how each one is built

Our identity is **experiments you can trust, not scores**. Each USP below is a concrete feature with an implementation path and a moment in the demo.

| # | USP | How it is implemented | Demo moment |
| --- | --- | --- | --- |
| 1 | **Two-level robustness: seeds and assumptions** | Run each scenario over S random seeds, and repeat over A samples of the uncertain parameters (tendencies, base rate, policy threshold) drawn from stated ranges. Report the share of the S × A runs where the variant beats the baseline | "Variant B wins in 87% of runs and in 9 of 10 assumption settings." A mentor asks "your coefficients are guesses" and the answer is on screen |
| 2 | **Cascade R decomposition** | Compute R, the new exposures generated per exposed viewer, per round and per community. Factor it as watch rate × share rate × reach per share × unseen-viewer rate, plus the recommendation-channel contribution. Every bottleneck names its weakest factor with numbers | "R is 0.62. The weak factor is share rate in the Food community." |
| 3 | **Reach attribution** | Tag every exposure with its channel (recommendation or share) and the source community and node role (hub, bridge, other). Draw a flow diagram | "62% of reach came from recommendation. Shares never left the seed community." |
| 4 | **Tipping-point curve** | Sweep early retention over a grid, run S seeds at each point, and plot the share of runs that reach a growth milestone. Mark the video's position | "You sit on the steep part of the curve, so a small hook gain moves survival a lot." |
| 5 | **Real-clip hook counterfactual** | The creator uploads an alternate opening. Both clips go through the same trained model on real frames, and the engine runs matched simulations | "Same audience, same graph, same dice; only the opening changed." |
| 6 | **Evidence ledger** | Every metric carries a tag: Learned, Calibrated, Assumed or Simulated, shown as a badge and listed on one screen | The trained vs assumed table, live |

### Why these matter for the judging criteria

- **Innovation:** the combination of cascade-survival analysis, assumption-robust verdicts and a real-clip counterfactual is not something I found in the audience simulators I reviewed (OASIS, MiroFish, MOSAIC). I have not checked every tool, so we claim it as "we did not find this elsewhere", not "first".
- **Technical complexity:** a trained model with held-out metrics, a vectorised engine with common random numbers, and a two-level Monte Carlo.
- **Demo:** the cascade replay, the tipping-point curve and the A/B verdict are all visual.
- **Impact and feasibility:** creators get an actionable answer ("what to change, and how sure are we"), and every limit is stated openly.
- **Pitch and Q&A:** the hard questions ("is it validated?", "are the parameters guesses?") already have on-screen answers.

## Data and model: a small SnapUGC chunk

We train on a small, fixed chunk of SnapUGC. It is enough to show a held-out result and cheap enough to finish in roughly 6 to 8 working hours (my estimate; confirm after timing a 200-video sample). From the [SnapUGC papers](https://arxiv.org/html/2509.02969v1) I read, the full dataset has 120,651 videos of 5 to 60 seconds, and the label is the Engagement Continuation Rate (ECR), the share of viewers who watch at least 5 seconds, rank-normalized. Your mirror may differ, so the actual labels must be inspected first.

### Access gate (hours 0 to 2, must pass)

1. List the splits and retrieve rows from the mirror.
2. Read the schema and confirm what each label field means.
3. Match video IDs to files and **download 5 to 10 raw MP4s**. Metadata access does not prove video access.
4. Confirm the videos decode and play.
5. Check the licence and access terms.
6. Fix the split handling and write a manifest so no video appears in two splits.
7. Time the download and frame extraction on 200 videos, then choose the chunk size.

If videos cannot be downloaded by hour 2, stop and switch plan before any other work.

### The chunk

| Split | Target size | Fallback if slow |
| --- | --- | --- |
| Train | 5,000 videos | 2,000 |
| Validation | 1,000 | 500 |
| Held-out test | 1,000 | 500 |

Use the official splits where they exist. Sample stratified by label quantile with a fixed seed, so every label range is represented, and save the video IDs in a manifest. These sizes are planning numbers; the timing test decides.

### Features (identical for training and inference)

- **Frames:** 8 uniformly sampled frames plus 4 extra frames from the first 3 seconds (the hook window).
- **Encoder:** one frozen, pretrained image-text encoder (a CLIP-style model). The same encoder later computes persona relevance, so one model serves two jobs.
- **Pooled features:** mean over all frames, mean over the hook window, the difference between them, and the temporal spread of frame embeddings.
- **Simple signals:** duration, cut count, motion energy, audio present, speech rate.
- Audio or transcript embeddings only if the baseline is stable.

### Model and training

1. Ridge regression on the pooled features, as the first baseline.
2. Gradient boosting or a small MLP; choose on validation.
3. Output: predicted label **percentile** in \[0, 1\], calibrated on validation.
4. Train an ensemble of 5 seeds. The spread of predictions is the **model uncertainty**, and the robustness loop reuses it.

No fine-tuning of a large video model. The published models needed [2 to 8 A800 GPUs](https://arxiv.org/html/2508.02516v2).

### Evaluation on the held-out chunk

| Report | Detail |
| --- | --- |
| SROCC and PLCC | The standard metrics in the literature |
| Pairwise ranking accuracy | 2,000 random pairs, plus the subset where labels differ by more than 0.2 |
| Baselines | random, duration only, simple signals only, frames without the hook window, full model |
| Hook ablation | Held-out score with and without the hook-window features |
| Reference points | Qwen2.5-VL fine-tuned 0.665, VideoLLaMA2 fine-tuned 0.691, challenge winner about 0.707 (SROCC; [source](https://arxiv.org/html/2508.02516v2), [challenge](https://arxiv.org/html/2509.02969v1)) |

With a small chunk and frozen features we should land below the published numbers. I would plan for roughly 0.35 to 0.55 SROCC (my estimate, not a measurement) and report whatever we measure. The hook ablation is a useful honest result: if the hook-window features improve the held-out score, that is correlational evidence that openings carry signal, without any causal claim.

### Model artifact

Save the weights, encoder name and version, frame sampling settings, preprocessing, feature normalisation, split manifest, metrics, software versions and a plain-language limitations note.

### Limitations

SnapUGC is Snapchat Spotlight content, not Instagram. The label is a relative percentile, so absolute probabilities in the simulation remain an assumption, and the robustness loop samples them.

## Video understanding and hook analysis

The video is analysed once and cached. A vision-language model writes the qualitative analysis, and the trained model plus embeddings supply every number that drives the simulation.

### Preprocessing

1. Validate the file type, decoding, duration, dimensions, frame rate and audio.
2. Extract the frames defined in the data section: 8 uniform and 4 from the first 3 seconds, with timestamps.
3. Extract audio and transcribe speech where present.
4. Detect scene cuts.
5. Cache all outputs by file hash.

Tools: FFmpeg and OpenCV for decoding, and PySceneDetect for cuts. For transcription use any speech-to-text tool that works offline or through an API; the choice is open.

### Qualitative analysis (one VLM call per video)

The model gets the sampled frames and transcript and returns validated JSON:

- Topic, category, subtopics, summary, format and narrative structure.
- Audience-interest tags, used for display and persona vocabulary.
- **Hook:** category (curiosity, bold claim, pain point, visual surprise, transformation, demonstration, mid-action start, direct address, humour, other or uncertain; more than one allowed), the opening window, strengths, weaknesses, evidence from frames or speech, a suggested improvement and a confidence value.

If the API fails, fall back to basic features plus a manual description, or a cached analysis for the demo video.

### What drives the simulation, and what does not

| Source | Role |
| --- | --- |
| Trained prior (percentile) | Drives round-0 attention |
| Frame embeddings | Drive persona relevance |
| VLM hook scores such as clarity or curiosity | **Display and the hypothetical-profile experiment only**, labelled "model-generated assessment" |

This reverses PRD v1.0, where LLM-generated hook numbers fed the engine directly. Keeping them out of the engine means the A/B difference comes from the trained model and real frames, not from numbers the LLM invented.

### Attention-risk timeline

Split the video into 8 windows. For each window, recompute the pooled features with that window masked and measure how much the trained prior changes. Windows with the largest change are flagged. This shows where the model is sensitive; it is not proof that viewers drop off there, and the UI says so.

### Alternate opening

The creator can upload a second clip with a different opening. It goes through the identical preprocessing and the same trained model, which is the input to the main counterfactual.

## Audience: archetypes and relevance

The audience is created once from LLM-written archetypes, sampled into thousands of viewers, and stored so that identities never change between videos. Different videos change relevance and behaviour, never who a viewer is.

### Archetypes (one-time LLM batch)

Generate 40 to 60 archetypes across 6 to 10 interest communities, covering many content categories, not only tech. Each archetype holds:

- A name and 3 to 6 short interest phrases with strengths.
- Trait ranges: attention tendency, novelty preference, and watch, like, comment, save and share tendencies.
- A community and an influence class.

Validate the schema and ranges, remove duplicates, and exclude sensitive traits such as ethnicity, religion, politics or health. Archetypes are fictional.

### Sampling the population

| Setting | Default | Note |
| --- | --- | --- |
| Viewers N | 4,000 (configurable, 1,000 to 5,000) | Large enough for hubs, bridges and depth to appear |
| Archetype choice | Weighted by community size | Weights are an assumption |
| Trait noise | Bounded noise around archetype values | Gives within-archetype variety |
| Influence | Heavy-tailed (a few hubs) | An assumption, stated in the ledger |
| Persistence | Parquet or SQLite, with a version and seed | Reused across videos |

I expect a few thousand viewers to be comfortable for a vectorised engine; confirm with the first prototype.

### Relevance from a joint embedding

1. Embed each viewer's interest phrases with the encoder's text side, then take the strength-weighted mean as the viewer's interest embedding.
2. Embed the video frames with the same encoder's image side.
3. Compute the cosine similarity of the two.
4. Normalise against a reference set of about 200 chunk videos: relevance is the viewer's percentile rank of that cosine, which spreads values over \[0, 1\] instead of clustering.

This is a zero-shot proxy, not a measurement of human interest. We check it with simple sanity tests, for example that a cooking video scores higher in the food community than in the gaming community, and we say in the UI that it is an approximation.

### Persona cards

The UI shows a handful of sample viewers (archetype, community, role, tendencies) so judges can see what the population is made of.

## Behaviour engine

Each exposed viewer goes through a short chain of probabilistic steps. The first step is anchored to the trained prior; everything after it is a transparent, tunable assumption. No LLM is called.

### The chain of steps

1. **Early attention (stay past about 5 seconds).** This is the only step tied to data.
2. **Watch fraction,** given the viewer stayed.
3. **Likes, comments and saves,** conditional on how much was watched.
4. **Share,** conditional on watching and on relevance.
5. **Log the event** with the dominant factor as a reason code.

Skipping, watching and sharing are therefore not independent: a viewer who skips cannot like or share.

### Formulas

The trained prior p (a percentile in \[0, 1\]) sets the base stay probability. The label is a rank, not a probability, so we map it between two assumed bounds:

```latex
q_0 = q_{\min} + (q_{\max} - q_{\min})\,p
```

```latex
P(\text{stay} \mid v) = \sigma\big(\mathrm{logit}(q_0) + a\,(\mathrm{rel}_v - \overline{\mathrm{rel}}) + c\,(\mathrm{attn}_v - 0.5)\big)
```

Watch fraction is drawn from a Beta distribution whose mean rises with relevance and attention and falls with duration. Like, comment, save and share probabilities are a base rate times the viewer's tendency times a function of watch fraction and relevance, clipped to \[0, 1\].

### Parameters and their status

| Parameter | Role | Status | Starting range (placeholder) |
| --- | --- | --- | --- |
| p | Video's engagement percentile | **Trained** | Model output, with ensemble spread |
| q\_min, q\_max | Maps percentile to stay probability | Assumed | 0.25 to 0.40 and 0.70 to 0.90 |
| rel | Relevance of the video to the viewer | **Computed** | From embeddings |
| a, c | Weight of relevance and attention | Assumed | 0.5 to 2.0 and 0.5 to 1.5 |
| Base like, comment, save rates | Engagement level | Assumed | Low single-digit percentages |
| Base share rate | Sharing level | Assumed | Below the like rate |
| Viewer tendencies | Individual differences | Assumed | From archetype ranges |

The starting ranges are my placeholders, not validated constants. Tune them by inspection so that a median-percentile video mostly gets skipped or partly watched, shares are rare, and no run saturates the whole population; write the chosen values down. In the robustness loop each uncertain parameter is **sampled from its range**, and the model's ensemble spread is sampled for p.

### Reason codes

For each decision, record which term contributed most (prior, relevance, attention or duration). The diagnosis uses these counts, so explanations come from logged evidence.

### Boundary stated in the product

The trained model gives a content-level prior. It does not predict any individual persona's behaviour, and the UI says that.

## Network and distribution

Viewers sit in a synthetic, community-structured follower graph. A video reaches them through two separate channels, recommendation and social sharing, and every exposure records which one.

### Graph

- **Model:** a directed, degree-corrected stochastic block model. Edges are denser inside a community than between communities, and influence sets how many followers a viewer has, so a few viewers become hubs.
- **Bridges:** a small share of viewers get extra cross-community edges.
- **Roles:** computed from the graph, not assigned. Hubs are the top 2% by follower count, and bridges are the top 3% by the share of edges that cross communities. These cut-offs are assumptions.
- **Build:** generate with NetworkX, then convert to a sparse adjacency matrix for the engine. Save with a version and seed.
- **Separate from personas,** so topology can change without regenerating identities.

The graph is synthetic, not Instagram's follower network, and hub and bridge effects are simulation assumptions.

### Channels

| Channel | How viewers are chosen | Logged as |
| --- | --- | --- |
| **Recommendation** | Each round the policy decides a batch size, then samples unseen viewers weighted by relevance, with an exploration share ε chosen at random | Channel = rec |
| **Social share** | A viewer who shares exposes some of their unseen followers, up to a fanout cap and an acceptance probability | Channel = share, with the sharing viewer as source |

### Distribution policy (ours, not Instagram's)

After each round the policy computes an expansion score from that round's audience:

```latex
S_t = \alpha\,\text{stay}_t + \beta\,\text{watchfrac}_t + \gamma\,\text{like}_t + \delta\,\text{share}_t, \qquad B_{t+1} = \mathrm{clip}\big(B_t\, e^{\kappa (S_t - S_0)},\, 0,\, B_{\max}\big)
```

The signal choice is inspired by [Instagram's public statement](https://www.socialmediatoday.com/news/instagram-shares-algorithm-insights-2025/738034/) that watch time, likes and sends matter most, but the rule is our own documented approximation. The weights, κ, S₀ and ε are assumptions with stated ranges.

Policy variants for experiments: **conservative** (higher S₀), **exploratory** (higher ε) and **interest-tight** (ε = 0).

### Seeding

| Strategy | Meaning |
| --- | --- |
| Random | Seed viewers drawn uniformly |
| Relevance-biased | Seed the most relevant viewers |
| Mixed | Half random, half relevance-biased |
| Community-targeted | Seed one chosen community |

Default seed audience is 1 to 2% of the population (about 40 to 80 viewers of 4,000). The strategy is recorded in every run and is one of the Lab's experiment variables.

### Stop conditions

Stop when no unseen viewer is eligible, when the expansion score stays below threshold for two rounds, when the maximum of 10 rounds is reached, or when the exposure budget is used. Re-exposure is off in the MVP.

## Simulation engine

The engine is a vectorised round loop over arrays, with a pre-drawn random tensor so that two scenarios can share exactly the same dice. It runs independently of the UI.

### State and round loop

State arrays per viewer: exposed, round exposed, channel, source viewer, stay, watch fraction, like, comment, save, share.

Each round:

1. Identify the newly exposed viewers.
2. Look up relevance and compute the behaviour probabilities as vectors.
3. Sample reactions using the pre-drawn uniforms.
4. Collect sharers and expose unseen followers through the share channel.
5. Compute the expansion score and the next batch size.
6. Select the recommendation batch.
7. Log events and update metrics, including R.
8. Check the stop conditions.

### Common random numbers

For each seed, draw one tensor of uniform numbers indexed by round, viewer and decision type (stay, watch fraction, like, comment, save, share, follower acceptance, recommendation tie-break). Baseline and variant both read the same tensor. A viewer therefore faces the same dice in both scenarios, and any difference comes from the changed input. For 10 rounds, 4,000 viewers and 8 decision types this is 320,000 numbers, about 1.3 MB as 32-bit floats.

If a scenario changes the audience selection, the viewers reached differ, but each viewer's own draws stay identical, which is the intended behaviour.

### Event log

Store columnar arrays and flush to Parquet or SQLite per run. Fields: run ID, round, viewer ID, event type, channel, source viewer, community, role, reason code and the relevant probability. Event types: exposure, skip, watch, partial view, completion, like, comment, save and share. The cascade must be reconstructable from the log alone.

### Reproducibility record

Every run stores the seed, the assumption-sample ID, the population, graph, model and policy versions, the video feature profile and all parameters. The same record reproduces the same run.

### Performance targets (estimates to confirm in the prototype)

| Job | Runs | Target |
| --- | --- | --- |
| One simulation, 4,000 viewers | 1 | Well under a second |
| Matched experiment, 30 to 50 seeds × 10 assumption samples × 2 scenarios | 600 to 1,000 | Under 2 minutes, with multiprocessing if needed |
| Tipping-point curve, 15 to 20 grid points × 30 seeds | 450 to 600 | Under 1 minute |

If the targets are missed, reduce the seeds and assumption samples first, and cache the demo runs.

## Propagation Lab

The Lab turns the simulation into an experiment. Every scenario is run many times under matched conditions, diagnosed with measurable evidence, and given a verdict together with how stable that verdict is.

### Experiment protocol

1. **Choose one variable:** the hook (a real alternate opening clip), the seed strategy, or the distribution policy.
2. **Hold everything else fixed:** population, graph, model version, policy (unless it is the variable), exposure budget and the random tensors.
3. **Outer loop, assumptions:** draw A assumption samples from the stated parameter ranges and from the model's ensemble spread.
4. **Inner loop, seeds:** for each assumption sample, run S seeds.
5. **Pair the runs:** baseline and variant use the identical (assumption sample, seed) pair and the same random tensor.
6. **Summarise** the paired differences on every metric.
7. **Save** the configuration record.

### Verdict and stability

| Output | Definition |
| --- | --- |
| Win rate | Share of paired runs where the variant has higher unique reach |
| Effect | Median paired difference with a 10th to 90th percentile interval |
| Assumption stability | Share of assumption samples where the variant's median beats the baseline's |
| Label | **Robust** if both rates are at least 80%, **Likely** if both are at least 60%, otherwise **Inconclusive** |
| Sensitivity | Rank correlation between each sampled parameter and the outcome, showing what the verdict depends on most |

The 80% and 60% cut-offs are our own labels, not statistical tests, and are described as such.

### Experiment record

Experiment ID, baseline and variant scenarios, changed and held-fixed variables, population, graph, model and policy versions, seeds, assumption samples, number of repetitions, parameters and results.

### Diagnosis with the reproduction number R

```latex
R_t = \frac{\text{new exposures generated by viewers exposed in round } t}{\text{viewers exposed in round } t}
```

The cascade grows when R is above 1. Split R into the recommendation part and the share part, and split the share part into measurable factors:

```latex
R_t^{\text{share}} = P(\text{stay})\cdot P(\text{share}\mid\text{watch})\cdot f_{\text{eff}}\cdot u\cdot \text{accept}
```

Here f\_eff is the average number of followers a sharer can reach, and u is the share of those followers not yet exposed. For each factor, compute the **uplift needed to bring R to 1 with the others held fixed**. The smallest required uplift names the binding constraint, for example "share rate would need to rise 1.6 times". This is computed per round and per community.

| Bottleneck | Measured evidence |
| --- | --- |
| Weak initial attention | Round-0 stay rate below its threshold |
| Low relevance in a newly reached community | That community's mean relevance below its threshold |
| Retention weakness | Mean watch fraction below its threshold |
| Insufficient sharing | Share-channel R far below 1, share rate low |
| Distribution threshold not reached | Expansion score below S₀ for two rounds |
| Community saturation | Over 80% of a community already exposed |
| Failure to cross communities | Cross-community share of exposures below its threshold |
| Lack of effective bridges | Few or no bridge viewers among sharers |
| Limited candidate audience | Eligible unseen viewers fewer than the batch size |
| Stochastic failure | The same scenario reaches the milestone in most other seeds |

Thresholds are starting assumptions, tuned during sanity testing. Every diagnosis line cites its numbers, and uses wording such as "in this simulated run".

### Pathway classification (computed from the log)

| Class | Rule (starting thresholds) |
| --- | --- |
| Early cascade failure | Ends by round 3 with reach under 5% of viewers |
| Niche, community-led | At least 70% of exposures in one community |
| Hub-driven | At least 40% of share-channel exposures come from hubs |
| Bridge-driven | At least 40% of cross-community exposures come from bridges |
| Cross-community | At least 60% of communities reach 10% penetration |
| Broad but shallow | Reach at least 30% with depth of 3 or less |
| Deep but narrow | Depth of 6 or more with reach under 15% |

A run can match several classes. The UI shows the primary class by the order above and lists the rest. These are descriptions of simulated outcomes, not validated viral categories.

### Reach attribution

A flow diagram built from the log: channel (recommendation or share), then source community, then viewer role (hub, bridge or other). It answers "where did the reach come from?".

### Tipping-point curve

Sweep the base stay probability across 15 to 20 grid points, run S seeds at each, and plot the share of runs that reach the growth milestone (default: 25% of viewers, an assumption). Mark where the video's predicted value sits. The curve shows how close the video is to the point where the cascade either dies out or takes off, and how much a hook improvement would shift it.

## Screens and demo script

The product has six screens, and the demo walks through them in one story: a video stalls, we show why, we test a better opening, and we show how sure we are.

### Screens

| # | Screen | Contents |
| --- | --- | --- |
| 1 | Upload and analysis | Preview, metadata, hook breakdown with evidence, the model's percentile with its caveat, the attention-risk timeline |
| 2 | Audience | Communities, sample persona cards, relevance heatmap by community |
| 3 | Propagation replay | Animated cascade by round, channel colouring, hubs and bridges, event inspector |
| 4 | Diagnosis | R by round and community, factor uplifts, the binding constraint, pathway class, reach attribution flow |
| 5 | Lab | Scenario setup, matched A/B, verdict with stability label, assumption sensitivity, tipping-point curve |
| 6 | Evidence ledger | The trained-vs-assumed table and the held-out model metrics with baselines |

UX rules: label every number with its evidence tag, show uncertainty next to every verdict, hide advanced controls by default, and keep the demo path to a few clicks.

### Demo script (about 4 minutes)

| Time | Screen | What happens | Fallback |
| --- | --- | --- | --- |
| 0:00 to 0:25 | Title | The problem: creators learn how a video lands only after posting | None |
| 0:25 to 1:00 | Upload and analysis | Upload a prepared video. Show the hook evidence and the model percentile. Say plainly that SnapUGC is a different platform | Cached analysis |
| 1:00 to 1:30 | Evidence ledger | Held-out metrics against baselines, and what is learned vs assumed | Pre-rendered screen |
| 1:30 to 2:10 | Replay and diagnosis | Run the baseline. The cascade stalls in one community. The diagnosis names the binding constraint with numbers and shows reach attribution | Precomputed run with a fixed seed |
| 2:10 to 3:05 | Lab | Upload the alternate opening. Run the matched experiment. Show win rate, effect interval and the stability label | Cached results |
| 3:05 to 3:30 | Tipping-point curve | Show the video's position and how the alternate opening moves it | Cached curve |
| 3:30 to 4:00 | Close | State the limits and the roadmap | None |

### Prepare before the day

- A demo video and an **alternate opening of the same video**, made in advance.
- Cached analysis, cached experiment results and a recorded backup of the full flow.
- A fixed seed for every run shown.
- Answers ready for: "Is this Instagram's algorithm?" (no, our documented policy), "How do you know it works?" (held-out metrics and the ledger), "Are the parameters guesses?" (assumed, and the verdict is tested across their ranges), "What did you not learn?" (persona-level behaviour).
- Check the dataset terms before showing any dataset video on screen.

## Modules, API and data contracts

Module boundaries matter more than API completeness. Agree the contracts below in hours 0 to 2 so four people can build in parallel against stubs.

### Stack

| Layer | Choice |
| --- | --- |
| Frontend | React with Vite (Streamlit only if nobody knows React; the cascade animation will be weaker) |
| Backend | Python with FastAPI |
| Video | FFmpeg, OpenCV, PySceneDetect |
| ML | PyTorch for the frozen encoder, scikit-learn or gradient boosting for the head |
| Engine | NumPy and SciPy sparse matrices; NetworkX for graph generation |
| Storage | SQLite and Parquet; persona and graph files with versions |

### Modules

| Module | Responsibility |
| --- | --- |
| ingestion, preprocessing | Validate, extract frames, audio, cuts; cache by hash |
| video\_analysis | VLM qualitative analysis and hook breakdown |
| engagement\_model | Features, ensemble, inference, held-out metrics |
| persona\_manager, audience\_matcher | Archetypes, sampling, persistence, relevance |
| graph\_manager | Graph generation, roles, metrics |
| behaviour\_model | Probabilities and sampled actions |
| distribution\_policy | Expansion score, batch size, selection |
| simulation\_engine, event\_logger | Round loop, common random numbers, logs |
| diagnostics | R decomposition, bottlenecks, pathways, attribution |
| experiment\_manager | Matched A/B, two-level robustness, tipping-point sweep |
| report\_generator | Wording from logged evidence only |
| api, frontend | Orchestration and UX |

### Endpoints (may be simplified)

- `POST /videos` upload, and `POST /videos/{id}/analyze` for preprocessing, analysis and inference.
- `GET /videos/{id}/analysis` for the result.
- `POST /simulations` and `GET /simulations/{run_id}` with `/events` and `/metrics`.
- `POST /experiments` and `GET /experiments/{id}`.
- `GET /ledger` for evidence tags and model metrics.
- `GET /health` for model status.

### Contracts

```json
{
  "video_profile": {
    "video_id": "...", "duration_s": 0, "hook_window_s": [0, 3],
    "percentile": 0.0, "percentile_ensemble": [0.0],
    "embedding_ref": "...", "vlm_analysis": {}, "model_version": "..."
  },
  "population": { "version": "...", "seed": 0, "viewers": "population.parquet", "graph": "graph.npz" },
  "run_config": {
    "run_id": "...", "seed": 0, "assumption_sample_id": 0,
    "seed_strategy": "random", "policy_version": "...", "params": {}
  },
  "event": {
    "run_id": "...", "round": 0, "viewer_id": 0, "type": "share",
    "channel": "share", "source_viewer": 0, "community": "...",
    "role": "hub", "reason": "relevance"
  },
  "experiment": {
    "experiment_id": "...", "baseline": {}, "variant": {},
    "changed": ["hook_clip"], "held_fixed": ["population", "graph", "policy"],
    "seeds": [0], "assumption_samples": 0, "results": {}
  }
}
```

These are field-level sketches to agree on, not final schemas.
