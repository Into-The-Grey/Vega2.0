import sys
from src.vega.core.db import set_memory_fact, get_memory_facts


def main():
    # Use None session (global)
    messy_key = "  User_Name\u200b\u200c  "
    value = "Alice"
    set_memory_fact(None, messy_key, value)
    facts = get_memory_facts(None)
    assert (
        "user_name" in facts
    ), f"Normalized key 'user_name' not found in facts: {facts.keys()}"
    assert facts["user_name"] == "Alice"

    # Another case: internal whitespace collapse and case
    messy_key2 = "  Preferred   TimeZone  "
    set_memory_fact(None, messy_key2, "UTC")
    facts2 = get_memory_facts(None)
    assert (
        "preferred timezone" in facts2
        or "preferred_timezone" in facts2
        or "preferred timezone" in facts2
    )
    # we used whitespace collapse+lowercase, underscore not added; check with space form
    assert (
        "preferred timezone" in facts2
    ), f"Expected 'preferred timezone' in facts: {facts2.keys()}"
    assert facts2["preferred timezone"] == "UTC"

    print("KEY NORMALIZATION TEST: OK")


if __name__ == "__main__":
    main()
