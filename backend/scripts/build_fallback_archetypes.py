"""Build backend/data/archetypes_fallback.json (hand-written, offline fallback).

Fictional archetypes only; interests are content topics, never sensitive traits.
Trait ranges are ASSUMED data, chosen by hand from five engagement profiles.
Run:  python scripts/build_fallback_archetypes.py
"""
import json
from pathlib import Path

# engagement profile -> trait (lo, hi). Tendencies are multipliers around 1.0.
PROFILES = {
    "lurker":   {"attention": (0.40, 0.60), "like": (0.5, 0.9), "comment": (0.3, 0.7), "save": (0.6, 1.0), "share": (0.4, 0.8)},
    "engager":  {"attention": (0.50, 0.70), "like": (1.1, 1.6), "comment": (1.1, 1.8), "save": (0.8, 1.2), "share": (0.8, 1.2)},
    "sharer":   {"attention": (0.45, 0.65), "like": (0.9, 1.3), "comment": (0.7, 1.1), "save": (0.7, 1.1), "share": (1.3, 2.0)},
    "focused":  {"attention": (0.60, 0.85), "like": (0.8, 1.2), "comment": (0.6, 1.0), "save": (1.2, 1.8), "share": (0.7, 1.1)},
    "scroller": {"attention": (0.25, 0.45), "like": (0.7, 1.1), "comment": (0.4, 0.8), "save": (0.4, 0.8), "share": (0.8, 1.3)},
}

# community -> [(name, profile, influence, [(phrase, strength), ...])]
COMMUNITIES = [
    ("Food & Cooking", [
        ("Home Cook", "engager", "medium", [("home cooking recipes", 0.9), ("weeknight dinners", 0.7), ("kitchen tips", 0.6)]),
        ("Baking Enthusiast", "focused", "medium", [("baking bread and cakes", 0.9), ("pastry techniques", 0.7), ("dessert decorating", 0.6)]),
        ("Street Food Explorer", "sharer", "high", [("street food", 0.9), ("food markets", 0.7), ("spicy food challenges", 0.5), ("travel eats", 0.4)]),
        ("Meal Prepper", "lurker", "low", [("meal prep", 0.9), ("high protein recipes", 0.7), ("grocery budgeting", 0.5)]),
        ("Restaurant Reviewer", "engager", "high", [("restaurant reviews", 0.9), ("fine dining", 0.6), ("food critique", 0.7)]),
        ("Vegan Recipe Fan", "sharer", "medium", [("vegan recipes", 0.9), ("plant based cooking", 0.8), ("easy salads", 0.5)]),
    ]),
    ("Gaming", [
        ("Competitive Gamer", "engager", "high", [("competitive esports", 0.9), ("fps gameplay", 0.8), ("ranked tips", 0.6)]),
        ("Cozy Gamer", "lurker", "medium", [("cozy farming games", 0.9), ("indie games", 0.7), ("relaxing gameplay", 0.6)]),
        ("Speedrun Watcher", "focused", "low", [("speedruns", 0.9), ("game glitches", 0.7), ("retro platformers", 0.5)]),
        ("Mobile Gamer", "scroller", "low", [("mobile games", 0.9), ("game memes", 0.6), ("gacha pulls", 0.6)]),
        ("Retro Collector", "focused", "medium", [("retro consoles", 0.9), ("game collecting", 0.8), ("arcade machines", 0.5)]),
        ("Game Lore Fan", "engager", "medium", [("video game lore", 0.9), ("story games", 0.7), ("fan theories", 0.7)]),
    ]),
    ("Fitness & Sport", [
        ("Gym Lifter", "engager", "high", [("weightlifting", 0.9), ("gym workouts", 0.8), ("muscle building tips", 0.7)]),
        ("Runner", "focused", "medium", [("running training", 0.9), ("marathon prep", 0.7), ("running shoes", 0.5)]),
        ("Yoga Practitioner", "lurker", "medium", [("yoga flows", 0.9), ("stretching routines", 0.7), ("mindful movement", 0.5)]),
        ("Football Fan", "sharer", "high", [("football highlights", 0.9), ("goal compilations", 0.8), ("match reactions", 0.6)]),
        ("Home Workout Beginner", "scroller", "low", [("home workouts", 0.9), ("beginner exercises", 0.8), ("quick routines", 0.6)]),
        ("Climber", "focused", "low", [("rock climbing", 0.9), ("bouldering", 0.8), ("outdoor adventure", 0.5)]),
    ]),
    ("Beauty & Fashion", [
        ("Skincare Routine Fan", "engager", "medium", [("skincare routines", 0.9), ("product reviews", 0.7), ("morning routines", 0.5)]),
        ("Makeup Artist", "focused", "high", [("makeup tutorials", 0.9), ("eyeliner techniques", 0.7), ("transformations", 0.7)]),
        ("Thrift Fashion Hunter", "sharer", "medium", [("thrift hauls", 0.9), ("outfit ideas", 0.8), ("sustainable fashion", 0.6)]),
        ("Streetwear Fan", "engager", "medium", [("streetwear outfits", 0.9), ("sneakers", 0.8), ("fashion drops", 0.6)]),
        ("Nail Art Lover", "lurker", "low", [("nail art", 0.9), ("manicure designs", 0.8), ("satisfying close ups", 0.5)]),
        ("Hair Styling Fan", "scroller", "low", [("hairstyle tutorials", 0.9), ("braids", 0.7), ("hair transformations", 0.6)]),
    ]),
    ("Tech & Gadgets", [
        ("Gadget Reviewer Fan", "engager", "high", [("gadget reviews", 0.9), ("unboxing", 0.8), ("smartphones", 0.7)]),
        ("Coder", "focused", "medium", [("programming tutorials", 0.9), ("software engineering", 0.7), ("coding tips", 0.7)]),
        ("PC Builder", "focused", "medium", [("pc building", 0.9), ("graphics cards", 0.7), ("desk setups", 0.7)]),
        ("AI Tools Explorer", "sharer", "high", [("ai tools", 0.9), ("productivity apps", 0.6), ("tech news", 0.7)]),
        ("Phone Photography Fan", "lurker", "low", [("phone photography", 0.9), ("photo editing", 0.7), ("camera tips", 0.6)]),
        ("Smart Home Tinkerer", "lurker", "low", [("smart home gadgets", 0.9), ("electronics projects", 0.7), ("home automation", 0.7)]),
    ]),
    ("Comedy & Entertainment", [
        ("Sketch Comedy Fan", "sharer", "high", [("sketch comedy", 0.9), ("funny skits", 0.8), ("relatable humour", 0.7)]),
        ("Meme Curator", "sharer", "medium", [("internet memes", 0.9), ("viral trends", 0.7), ("reaction clips", 0.6)]),
        ("Pet Video Lover", "scroller", "medium", [("funny pets", 0.9), ("cute cats and dogs", 0.9), ("animal videos", 0.7)]),
        ("Prank Watcher", "scroller", "low", [("pranks", 0.9), ("hidden camera", 0.6), ("challenges", 0.6)]),
        ("Movie Buff", "focused", "medium", [("movie reviews", 0.9), ("film trivia", 0.7), ("trailers", 0.6)]),
        ("Music Discovery Fan", "engager", "medium", [("new music", 0.9), ("song covers", 0.7), ("live performances", 0.6)]),
        ("Dance Trend Follower", "scroller", "high", [("dance trends", 0.9), ("choreography", 0.7), ("viral songs", 0.6)]),
    ]),
    ("Travel & Outdoors", [
        ("Budget Backpacker", "sharer", "medium", [("budget travel", 0.9), ("backpacking", 0.8), ("hostels", 0.5)]),
        ("Van Life Dreamer", "lurker", "low", [("van life", 0.9), ("road trips", 0.8), ("camping", 0.6)]),
        ("Hiker", "focused", "low", [("hiking trails", 0.9), ("mountains", 0.8), ("nature views", 0.7)]),
        ("Luxury Travel Fan", "engager", "high", [("luxury hotels", 0.9), ("resorts", 0.7), ("first class travel", 0.6)]),
        ("Beach Lover", "scroller", "medium", [("beaches", 0.9), ("islands", 0.8), ("snorkelling", 0.5)]),
        ("City Explorer", "engager", "medium", [("city guides", 0.9), ("hidden gems", 0.8), ("architecture", 0.6)]),
    ]),
    ("Learning & DIY", [
        ("Science Explainer Fan", "focused", "high", [("science explainers", 0.9), ("space and physics", 0.8), ("experiments", 0.7)]),
        ("History Buff", "focused", "medium", [("history stories", 0.9), ("ancient civilisations", 0.7), ("documentaries", 0.6)]),
        ("Personal Finance Learner", "lurker", "medium", [("personal finance", 0.9), ("saving money", 0.8), ("budgeting tips", 0.7)]),
        ("DIY Home Fixer", "engager", "medium", [("diy home repair", 0.9), ("home renovation", 0.8), ("tool tips", 0.6)]),
        ("Language Learner", "lurker", "low", [("language learning", 0.9), ("vocabulary tips", 0.7), ("culture facts", 0.5)]),
        ("Woodworker", "focused", "low", [("woodworking", 0.9), ("furniture making", 0.8), ("satisfying craftsmanship", 0.6)]),
        ("Productivity Seeker", "sharer", "medium", [("productivity hacks", 0.9), ("study tips", 0.8), ("life advice", 0.6)]),
    ]),
]


def build() -> list[dict]:
    out = []
    for cid, (cname, members) in enumerate(COMMUNITIES):
        for name, profile, influence, interests in members:
            out.append({
                "name": name,
                "community": cid,
                "community_name": cname,
                "interests": [{"phrase": p, "strength": s} for p, s in interests],
                "traits": {k: list(v) for k, v in PROFILES[profile].items()},
                "influence_class": influence,
            })
    return out


if __name__ == "__main__":
    data = build()
    assert len(data) == 50, len(data)
    root = Path(__file__).resolve().parent.parent / "data"
    for fname in ("archetypes_fallback.json", "archetypes.json"):
        (root / fname).write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {len(data)} archetypes")
