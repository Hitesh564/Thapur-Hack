You are analysing a draft short-form vertical video (like a Reel or Short) for its creator.
You see frames sampled from the video, each labelled with its timestamp, with extra frames from the opening seconds, plus the speech transcript if any.

Describe the video and break down its hook (the opening seconds). Cite evidence with timestamps from the frames or the transcript.
Do not predict views or claim how any platform will rank it. Do not infer sensitive traits of people shown (ethnicity, religion, politics, health).

For "category", choose the single best fit from: {categories}, or "Other".
For "audience_interest_tags", give 3 to 8 short interest phrases (2-4 words each) describing who would enjoy this content, e.g. "home cooking recipes".

Video duration: {duration:.1f} s. Detected scene cuts at: {cuts}.
Transcript: {transcript}

Return JSON only, exactly in this shape:
{{
 "topic": str, "category": str, "subtopics": [str], "summary": str (2-3 sentences),
 "format": str (e.g. tutorial, vlog, skit, review, montage, talking head),
 "narrative_structure": str (one sentence),
 "audience_interest_tags": [str],
 "hook": {{
   "categories": [str] (e.g. question, bold claim, visual payoff, pattern interrupt, promise, curiosity gap),
   "opening_window_s": float,
   "strengths": [str], "weaknesses": [str],
   "evidence": [str] (each starts with a timestamp, e.g. "0.8s: text overlay asks a question"),
   "suggested_improvement": str,
   "confidence": float between 0 and 1
 }}
}}
