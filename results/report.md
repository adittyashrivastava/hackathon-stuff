# Arjun's Memory — HydraDB Eval Report

**242 questions** across 10 BEAM-inspired categories and 10 dataset/collection scopes. **Overall mean score: 52.31%**

## By BEAM category

| category | N | Mean score | correct | partial | incorrect | hallucinated | error |
|---|---|---|---|---|---|---|---|
| abstention | 24 | 0.80 | 18 | 2 | 1 | 3 | 0 |
| contradiction_resolution | 24 | 0.67 | 15 | 2 | 7 | 0 | 0 |
| event_ordering | 23 | 0.35 | 7 | 1 | 11 | 4 | 0 |
| information_extraction | 25 | 0.57 | 12 | 3 | 9 | 1 | 0 |
| instruction_following | 20 | 0.59 | 10 | 3 | 6 | 1 | 0 |
| knowledge_update | 24 | 0.30 | 5 | 1 | 17 | 0 | 1 |
| multi_session_reasoning | 30 | 0.52 | 11 | 6 | 11 | 2 | 0 |
| preference_following | 24 | 0.48 | 11 | 1 | 10 | 2 | 0 |
| summarization | 24 | 0.57 | 10 | 3 | 7 | 3 | 1 |
| temporal_reasoning | 24 | 0.38 | 8 | 0 | 13 | 3 | 0 |

## By dataset / relationship

| dataset | N | Mean score | correct | partial | incorrect | hallucinated | error |
|---|---|---|---|---|---|---|---|
| friend_group | 53 | 0.35 | 13 | 5 | 32 | 2 | 1 |
| friend_group+gopal | 5 | 0.46 | 2 | 0 | 2 | 1 | 0 |
| friend_group+mother_son | 5 | 0.56 | 2 | 1 | 2 | 0 | 0 |
| friend_group+prof_jatin | 5 | 0.42 | 1 | 2 | 1 | 1 | 0 |
| gopal | 54 | 0.53 | 28 | 1 | 22 | 3 | 0 |
| gopal+mother_son | 5 | 0.64 | 3 | 0 | 2 | 0 | 0 |
| gopal+prof_jatin | 5 | 0.54 | 2 | 1 | 2 | 0 | 0 |
| mother_son | 53 | 0.68 | 30 | 7 | 12 | 3 | 1 |
| mother_son+prof_jatin | 5 | 0.52 | 1 | 2 | 2 | 0 | 0 |
| prof_jatin | 52 | 0.52 | 25 | 3 | 15 | 9 | 0 |
