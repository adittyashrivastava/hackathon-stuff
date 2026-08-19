# Arjun's Memory — HydraDB Eval Report

**242 questions** across 10 BEAM-inspired categories and 10 dataset/collection scopes. **Overall mean score: 48.41%**

## By BEAM category

| category | N | Mean score | correct | partial | incorrect | hallucinated | error |
|---|---|---|---|---|---|---|---|
| abstention | 24 | 0.80 | 19 | 0 | 3 | 2 | 0 |
| contradiction_resolution | 24 | 0.58 | 13 | 1 | 10 | 0 | 0 |
| event_ordering | 23 | 0.43 | 9 | 1 | 8 | 5 | 0 |
| information_extraction | 25 | 0.48 | 9 | 5 | 9 | 2 | 0 |
| instruction_following | 20 | 0.42 | 7 | 2 | 11 | 0 | 0 |
| knowledge_update | 24 | 0.33 | 6 | 3 | 15 | 0 | 0 |
| multi_session_reasoning | 30 | 0.44 | 8 | 7 | 12 | 3 | 0 |
| preference_following | 24 | 0.57 | 12 | 2 | 9 | 1 | 0 |
| summarization | 24 | 0.49 | 7 | 6 | 9 | 2 | 0 |
| temporal_reasoning | 24 | 0.32 | 7 | 1 | 12 | 4 | 0 |

## By dataset / relationship

| dataset | N | Mean score | correct | partial | incorrect | hallucinated | error |
|---|---|---|---|---|---|---|---|
| friend_group | 53 | 0.33 | 15 | 2 | 32 | 4 | 0 |
| friend_group+gopal | 5 | 0.32 | 1 | 1 | 2 | 1 | 0 |
| friend_group+mother_son | 5 | 0.66 | 2 | 2 | 1 | 0 | 0 |
| friend_group+prof_jatin | 5 | 0.28 | 0 | 2 | 2 | 1 | 0 |
| gopal | 54 | 0.44 | 21 | 4 | 25 | 4 | 0 |
| gopal+mother_son | 5 | 0.40 | 1 | 1 | 3 | 0 | 0 |
| gopal+prof_jatin | 5 | 0.60 | 3 | 0 | 1 | 1 | 0 |
| mother_son | 53 | 0.64 | 30 | 6 | 13 | 4 | 0 |
| mother_son+prof_jatin | 5 | 0.36 | 1 | 1 | 3 | 0 | 0 |
| prof_jatin | 52 | 0.55 | 23 | 9 | 16 | 4 | 0 |
