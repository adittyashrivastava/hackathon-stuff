# Arjun's Memory — HydraDB Eval Report

**242 questions** across 10 BEAM-inspired categories and 10 dataset/collection scopes. **Overall mean score: 52.52%**

## By BEAM category

| category | N | Mean score | correct | partial | incorrect | hallucinated | error |
|---|---|---|---|---|---|---|---|
| abstention | 24 | 0.90 | 21 | 1 | 0 | 2 | 0 |
| contradiction_resolution | 24 | 0.57 | 13 | 1 | 9 | 1 | 0 |
| event_ordering | 23 | 0.40 | 8 | 2 | 9 | 4 | 0 |
| information_extraction | 25 | 0.61 | 13 | 3 | 8 | 1 | 0 |
| instruction_following | 20 | 0.42 | 6 | 3 | 10 | 1 | 0 |
| knowledge_update | 24 | 0.33 | 6 | 2 | 15 | 1 | 0 |
| multi_session_reasoning | 30 | 0.54 | 10 | 9 | 11 | 0 | 0 |
| preference_following | 24 | 0.61 | 14 | 2 | 6 | 2 | 0 |
| summarization | 24 | 0.45 | 7 | 5 | 5 | 7 | 0 |
| temporal_reasoning | 24 | 0.38 | 8 | 2 | 12 | 2 | 0 |

## By dataset / relationship

| dataset | N | Mean score | correct | partial | incorrect | hallucinated | error |
|---|---|---|---|---|---|---|---|
| friend_group | 53 | 0.39 | 17 | 6 | 24 | 6 | 0 |
| friend_group+gopal | 5 | 0.64 | 2 | 2 | 1 | 0 | 0 |
| friend_group+mother_son | 5 | 0.54 | 2 | 1 | 2 | 0 | 0 |
| friend_group+prof_jatin | 5 | 0.42 | 1 | 2 | 2 | 0 | 0 |
| gopal | 54 | 0.44 | 21 | 4 | 26 | 3 | 0 |
| gopal+mother_son | 5 | 0.84 | 3 | 2 | 0 | 0 | 0 |
| gopal+prof_jatin | 5 | 0.48 | 1 | 2 | 2 | 0 | 0 |
| mother_son | 53 | 0.71 | 34 | 4 | 9 | 6 | 0 |
| mother_son+prof_jatin | 5 | 0.32 | 1 | 0 | 4 | 0 | 0 |
| prof_jatin | 52 | 0.56 | 24 | 7 | 15 | 6 | 0 |
