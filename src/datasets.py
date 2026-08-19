"""Shared registry of the four Arjun relationship-thread datasets."""

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATABASE = "arjun"

DATASETS = {
    "friend_group": {
        "file": "arjun_core5_friend_group_2800_messages.jsonl",
        "collection": "friend_group",
        "ground_truth_file": "arjun_core5_friend_group_ground_truth.json",
        "ground_truth_is_authored": True,
        "relationship": "Arjun's five-person school-friend group (Gopal, Priyanka, Rahul, Akhil)",
    },
    "gopal": {
        "file": "arjun_gopal_10year_2500_messages.jsonl",
        "collection": "gopal",
        "ground_truth_file": "arjun_gopal_memory_ground_truth_v2.json",
        "ground_truth_is_authored": True,
        "relationship": "Arjun's 1:1 best-friend WhatsApp thread with Gopal",
    },
    "mother_son": {
        "file": "mother_son_whatsapp_1year_synthetic.jsonl",
        "collection": "mother_son",
        "ground_truth_file": "mother_son_ground_truth_derived.json",
        "ground_truth_is_authored": False,
        "relationship": "Arjun's WhatsApp thread with his mother",
    },
    "prof_jatin": {
        "file": "arjun_prof_jatin_2000_messages.jsonl",
        "collection": "prof_jatin",
        "ground_truth_file": "prof_jatin_ground_truth_derived.json",
        "ground_truth_is_authored": False,
        "relationship": "Arjun's mentor/mentee thread with Prof. Jatin",
    },
}


def dataset_path(name: str) -> str:
    return os.path.join(DATA_DIR, DATASETS[name]["file"])


def ground_truth_path(name: str) -> str:
    return os.path.join(DATA_DIR, DATASETS[name]["ground_truth_file"])
