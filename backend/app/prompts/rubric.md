You are rating a draft short-form vertical video (like a Reel or Short) as a short-form video, before it is published.
You see frames sampled from the video, each labelled with its timestamp, with extra frames from the opening seconds, plus the speech transcript if any.

Score each dimension as an integer from {rmin} to {rmax}:
- immediate_clarity: within the first 1-2 seconds, is it obvious what the video is about?
- curiosity: does the opening create a reason to keep watching (a question, tension, a promise)?
- visual_impact: are the opening visuals striking, well lit, well framed, with movement or contrast?
- pacing: does the video move at a good speed for short-form (cuts, progression, no dead time)?
- payoff_clarity: is there a clear payoff or result that rewards watching to the end?
- audience_fit: how well does the content suit a short-form audience that is interested in this topic?

Be calibrated: 5 is an ordinary amateur video, 8+ is reserved for clearly excellent work, 3 or less for clearly weak work.
Judge only what you can see and hear in the material given. Do not guess view counts.

Video duration: {duration:.1f} s. Detected scene cuts: {n_cuts}.
Transcript: {transcript}

Return JSON only, exactly in this shape:
{{"scores": {{"immediate_clarity": int, "curiosity": int, "visual_impact": int, "pacing": int, "payoff_clarity": int, "audience_fit": int}}, "rationale": "one or two sentences"}}
