"""
Corner case tests for _extract_balanced_batch logic.
Standalone script — no dependency on verl/ray, just numpy.
"""
import numpy as np

# ============================================================
# Minimal mock classes to simulate DataProto and config
# ============================================================
class MockDataProto:
    def __init__(self, scores):
        self.non_tensor_batch = {"score": np.array(scores, dtype=float)}
        self._n = len(scores)

    def select_idxs(self, idxs):
        new = MockDataProto(self.non_tensor_batch["score"][idxs])
        return new

    def __len__(self):
        return self._n


class MockConfig:
    def __init__(self, ratio_high=5/8):
        self._ratio_high = ratio_high

    def get(self, key, default=None):
        if key == "ratio_buffer_high":
            return self._ratio_high
        return default


# ============================================================
# Extracted function (mirrors ray_trainer.py logic exactly)
# ============================================================
def extract_balanced_batch(config_trainer, merged_buffer, target_batch_size):
    scores = np.array(merged_buffer.non_tensor_batch['score'])
    pos_indices = np.where(scores > 0)[0]
    neg_indices = np.where(scores <= 0)[0]
    n_pos, n_neg = len(pos_indices), len(neg_indices)

    ratio_high = config_trainer.get("ratio_buffer_high", 5/8)
    max_allowed = int(np.floor(target_batch_size * ratio_high))

    info = {"buffer/original_pos": n_pos, "buffer/original_neg": n_neg}

    if n_pos == 0 or n_neg == 0:
        info["buffer/balanced_batch_size"] = n_pos + n_neg
        info["buffer/minority_class"] = "none"
        info["buffer/pos_take"] = n_pos
        info["buffer/neg_take"] = n_neg
        return merged_buffer, info

    if n_pos <= n_neg:
        minority_indices, majority_indices = pos_indices, neg_indices
        minority_label = "pos"
    else:
        minority_indices, majority_indices = neg_indices, pos_indices
        minority_label = "neg"

    n_minority_take = min(len(minority_indices), max_allowed)
    n_majority_take = target_batch_size - n_minority_take

    if n_majority_take > len(majority_indices):
        n_majority_take = len(majority_indices)

    rng = np.random.default_rng(42)
    if n_minority_take < len(minority_indices):
        sampled_minority = rng.choice(minority_indices, size=n_minority_take, replace=False)
    else:
        sampled_minority = minority_indices
    sampled_majority = rng.choice(majority_indices, size=n_majority_take, replace=False)

    selected = np.concatenate([sampled_minority, sampled_majority])

    balanced_batch = merged_buffer.select_idxs(selected.tolist())
    info["buffer/balanced_batch_size"] = len(selected)
    info["buffer/minority_class"] = minority_label
    info["buffer/minority_take"] = n_minority_take
    info["buffer/majority_take"] = n_majority_take
    info["buffer/pos_take"] = n_minority_take if minority_label == "pos" else n_majority_take
    info["buffer/neg_take"] = n_majority_take if minority_label == "pos" else n_minority_take
    return balanced_batch, info


# ============================================================
# Test cases
# ============================================================
TARGET = 1024
RATIO_HIGH = 5 / 8  # max_allowed = floor(1024 * 0.625) = 640
MAX_ALLOWED = int(np.floor(TARGET * RATIO_HIGH))  # 640
RATIO_LOW = 3 / 8   # min_required = ceil(1024 * 0.375) = 384
MIN_REQUIRED = int(np.ceil(TARGET * RATIO_LOW))    # 384

config = MockConfig(ratio_high=RATIO_HIGH)

def run_test(name, pos, neg, expected_pos_take, expected_neg_take, expected_batch_size=None):
    if expected_batch_size is None:
        expected_batch_size = expected_pos_take + expected_neg_take
    scores = [1.0] * pos + [0.0] * neg
    buf = MockDataProto(scores)
    result, info = extract_balanced_batch(config, buf, TARGET)
    actual_pos_take = info["buffer/pos_take"]
    actual_neg_take = info["buffer/neg_take"]
    actual_batch_size = info["buffer/balanced_batch_size"]

    passed = (
        actual_pos_take == expected_pos_take
        and actual_neg_take == expected_neg_take
        and actual_batch_size == expected_batch_size
    )
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    print(f"  Input: pos={pos}, neg={neg}, target={TARGET}, max_allowed={MAX_ALLOWED}")
    print(f"  Expected: pos_take={expected_pos_take}, neg_take={expected_neg_take}, batch_size={expected_batch_size}")
    print(f"  Actual:   pos_take={actual_pos_take}, neg_take={actual_neg_take}, batch_size={actual_batch_size}")
    if not passed:
        print(f"  >>> MISMATCH <<<")
    print()
    return passed


results = []

# --- Case 1: Normal case, minority < max_allowed ---
# pos=400 (minority), neg=800 (majority)
# minority_take = min(400, 640) = 400, majority_take = 1024 - 400 = 624
results.append(run_test("Normal: pos minority, both sufficient",
    pos=400, neg=800,
    expected_pos_take=400, expected_neg_take=624))

# --- Case 2: Minority exactly at max_allowed ---
# pos=640 (minority since 640 <= 2000), neg=2000
# minority_take = min(640, 640) = 640, majority_take = 1024 - 640 = 384
results.append(run_test("Minority exactly at max_allowed",
    pos=640, neg=2000,
    expected_pos_take=640, expected_neg_take=384))

# --- Case 3: Minority exceeds max_allowed (the bug case) ---
# pos=2634 (minority), neg=6582
# minority_take = min(2634, 640) = 640, majority_take = 1024 - 640 = 384
results.append(run_test("Minority exceeds max_allowed (original bug)",
    pos=2634, neg=6582,
    expected_pos_take=640, expected_neg_take=384))

# --- Case 4: Both sides very large ---
# pos=5000, neg=5000
# minority=pos (tie goes to pos), minority_take = min(5000, 640) = 640
# majority_take = 1024 - 640 = 384
results.append(run_test("Both sides very large, equal",
    pos=5000, neg=5000,
    expected_pos_take=640, expected_neg_take=384))

# --- Case 5: Neg is minority ---
# pos=800, neg=300
# minority=neg, minority_take = min(300, 640) = 300
# majority_take = 1024 - 300 = 724
results.append(run_test("Neg is minority, below max_allowed",
    pos=800, neg=300,
    expected_pos_take=724, expected_neg_take=300))

# --- Case 6: All positive (edge case) ---
# pos=1024, neg=0 -> no balancing, return full buffer
results.append(run_test("All positive, no negatives",
    pos=1024, neg=0,
    expected_pos_take=1024, expected_neg_take=0))

# --- Case 7: All negative (edge case) ---
results.append(run_test("All negative, no positives",
    pos=0, neg=1024,
    expected_pos_take=0, expected_neg_take=1024))

# --- Case 8: Minority very small (below min_required) ---
# pos=50 (minority), neg=2000
# minority_take = min(50, 640) = 50, majority_take = 1024 - 50 = 974
# Note: this would be a forced update (below ratio_low), but extraction still works
results.append(run_test("Minority very small (forced update scenario)",
    pos=50, neg=2000,
    expected_pos_take=50, expected_neg_take=974))

# --- Case 9: Majority not enough to fill ---
# pos=100 (minority), neg=500
# minority_take = min(100, 640) = 100, majority_take = 1024 - 100 = 924
# But neg only has 500, so majority_take capped to 500
results.append(run_test("Majority not enough to fill target",
    pos=100, neg=500,
    expected_pos_take=100, expected_neg_take=500,
    expected_batch_size=600))

# --- Case 10: Exactly target_batch_size total samples, 50/50 ---
# pos=512, neg=512
# minority=pos, minority_take = min(512, 640) = 512
# majority_take = 1024 - 512 = 512
results.append(run_test("Exactly target total, 50/50 split",
    pos=512, neg=512,
    expected_pos_take=512, expected_neg_take=512))

# --- Case 11: Single positive, many negatives ---
results.append(run_test("Single positive sample",
    pos=1, neg=5000,
    expected_pos_take=1, expected_neg_take=1023))

# --- Case 12: Single negative, many positives ---
# minority=neg(1), minority_take=min(1,640)=1, majority_take=1024-1=1023
results.append(run_test("Single negative sample",
    pos=5000, neg=1,
    expected_pos_take=1023, expected_neg_take=1))

# ============================================================
# Summary
# ============================================================
n_pass = sum(results)
n_total = len(results)
print("=" * 60)
print(f"Extraction Results: {n_pass}/{n_total} passed")
if n_pass == n_total:
    print("All extraction tests passed!")
else:
    print(f"{n_total - n_pass} extraction test(s) FAILED")


# ============================================================
# Part 2: can_fill_batch (should_update trigger) tests
# ============================================================
print("\n" + "=" * 60)
print("Part 2: can_fill_batch / should_update trigger tests")
print("=" * 60 + "\n")

def can_fill_batch(pos_count, neg_count, target_batch_size=TARGET,
                   ratio_low=RATIO_LOW, ratio_high=RATIO_HIGH):
    """Mirrors the can_fill_batch logic in ray_trainer.py fit()"""
    n_minority = min(pos_count, neg_count)
    n_majority = max(pos_count, neg_count)
    min_required = int(np.ceil(target_batch_size * ratio_low))    # 384
    max_allowed = int(np.floor(target_batch_size * ratio_high))   # 640
    result = (
        n_minority >= min_required
        and n_majority >= (target_batch_size - max_allowed)
    )
    return result, min_required, max_allowed


def should_update(pos_count, neg_count, buffer_step_count, is_last_step=False,
                  max_buffer_steps=10):
    """Mirrors should_update logic"""
    cfb, _, _ = can_fill_batch(pos_count, neg_count)
    return cfb or (buffer_step_count >= max_buffer_steps) or is_last_step


def run_update_test(name, pos, neg, buffer_steps, is_last, expected_update,
                    expected_reason="", max_buffer_steps=10):
    actual = should_update(pos, neg, buffer_steps, is_last, max_buffer_steps)
    cfb, min_req, max_alw = can_fill_batch(pos, neg)
    passed = actual == expected_update
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    print(f"  Input: pos={pos}, neg={neg}, buffer_steps={buffer_steps}, "
          f"is_last={is_last}, max_buffer_steps={max_buffer_steps}")
    print(f"  can_fill_batch={cfb} (min_required={min_req}, max_allowed={max_alw})")
    print(f"  Expected update={expected_update}, Actual update={actual}")
    if expected_reason:
        print(f"  Reason: {expected_reason}")
    if not passed:
        print(f"  >>> MISMATCH <<<")
    print()
    return passed


results2 = []

# --- U1: Normal trigger — minority in [min_req, max_allowed], majority enough ---
# pos=400, neg=800 -> minority=400 >= 384, majority=800 >= 384 -> True
results2.append(run_update_test("Normal: ratio satisfied",
    pos=400, neg=800, buffer_steps=1, is_last=False,
    expected_update=True,
    expected_reason="minority=400 >= min_required=384, majority=800 >= 384"))

# --- U2: Minority just below min_required -> SKIP ---
# pos=383, neg=2000 -> minority=383 < 384 -> False
results2.append(run_update_test("Minority just below min_required",
    pos=383, neg=2000, buffer_steps=1, is_last=False,
    expected_update=False,
    expected_reason="minority=383 < min_required=384"))

# --- U3: Minority exactly at min_required -> UPDATE ---
# pos=384, neg=2000 -> minority=384 >= 384 -> True
results2.append(run_update_test("Minority exactly at min_required",
    pos=384, neg=2000, buffer_steps=1, is_last=False,
    expected_update=True,
    expected_reason="minority=384 == min_required=384"))

# --- U4: Original bug case — minority far exceeds max_allowed ---
# pos=2634, neg=6582 -> minority=2634 >= 384, majority=6582 >= 384 -> True
results2.append(run_update_test("Original bug: minority >> max_allowed (should now trigger)",
    pos=2634, neg=6582, buffer_steps=9, is_last=False,
    expected_update=True,
    expected_reason="minority=2634 >= 384, majority=6582 >= 384"))

# --- U5: Both sides large and equal ---
# pos=5000, neg=5000 -> minority=5000 >= 384 -> True
results2.append(run_update_test("Both sides very large, equal",
    pos=5000, neg=5000, buffer_steps=1, is_last=False,
    expected_update=True))

# --- U6: All positive -> SKIP (minority=0 < 384) ---
results2.append(run_update_test("All positive, no negatives",
    pos=1024, neg=0, buffer_steps=1, is_last=False,
    expected_update=False,
    expected_reason="minority=0 < min_required=384"))

# --- U7: All negative -> SKIP ---
results2.append(run_update_test("All negative, no positives",
    pos=0, neg=1024, buffer_steps=1, is_last=False,
    expected_update=False,
    expected_reason="minority=0 < min_required=384"))

# --- U8: All positive but max_buffer_steps reached -> FORCED UPDATE ---
results2.append(run_update_test("All positive, forced by max_buffer_steps",
    pos=1024, neg=0, buffer_steps=10, is_last=False,
    expected_update=True,
    expected_reason="can_fill_batch=False but buffer_steps >= max_buffer_steps"))

# --- U9: All negative, forced by is_last_step ---
results2.append(run_update_test("All negative, forced by is_last_step",
    pos=0, neg=1024, buffer_steps=3, is_last=True,
    expected_update=True,
    expected_reason="can_fill_batch=False but is_last_step=True"))

# --- U10: Very few positive, not enough for ratio ---
# pos=50, neg=2000 -> minority=50 < 384 -> False
results2.append(run_update_test("Very few positives, ratio not met",
    pos=50, neg=2000, buffer_steps=3, is_last=False,
    expected_update=False,
    expected_reason="minority=50 < min_required=384"))

# --- U11: Majority not enough ---
# pos=500, neg=300 -> minority=300 < 384 -> False
results2.append(run_update_test("Minority below threshold (neg=300)",
    pos=500, neg=300, buffer_steps=2, is_last=False,
    expected_update=False,
    expected_reason="minority=300 < min_required=384"))

# --- U12: Majority barely enough ---
# pos=384, neg=384 -> minority=384 >= 384, majority=384 >= 384 -> True
results2.append(run_update_test("Both sides exactly at min_required",
    pos=384, neg=384, buffer_steps=1, is_last=False,
    expected_update=True,
    expected_reason="minority=384 >= 384, majority=384 >= 384"))

# --- U13: First step of real experiment (from log) ---
# pos=59, neg=965 -> minority=59 < 384 -> False
results2.append(run_update_test("Real log: step 1 (pos=59, neg=965)",
    pos=59, neg=965, buffer_steps=1, is_last=False,
    expected_update=False,
    expected_reason="minority=59 < min_required=384"))

# --- U14: Majority exists but too few to fill ---
# pos=500 (minority), neg=383 -> minority=383 < 384 -> False
# Wait, minority = min(500,383) = 383
results2.append(run_update_test("Majority too few (neg=383 is minority)",
    pos=500, neg=383, buffer_steps=1, is_last=False,
    expected_update=False,
    expected_reason="minority=383 < min_required=384"))

# --- U15: Edge — minority at max_allowed, majority barely enough ---
# pos=640, neg=384 -> minority=384 >= 384, majority=640 >= 384 -> True
results2.append(run_update_test("Minority=384, majority=640",
    pos=640, neg=384, buffer_steps=1, is_last=False,
    expected_update=True))

# --- U16: buffer_steps exactly at max-1, not triggered ---
results2.append(run_update_test("Buffer steps at max-1, ratio not met",
    pos=100, neg=2000, buffer_steps=9, is_last=False,
    expected_update=False,
    expected_reason="minority=100 < 384, buffer_steps=9 < 10"))

# --- U17: buffer_steps exactly at max, forced ---
results2.append(run_update_test("Buffer steps at max, forced update",
    pos=100, neg=2000, buffer_steps=10, is_last=False,
    expected_update=True,
    expected_reason="buffer_steps=10 >= max_buffer_steps=10"))

# ============================================================
# Final Summary
# ============================================================
n_pass2 = sum(results2)
n_total2 = len(results2)
print("=" * 60)
print(f"Update Trigger Results: {n_pass2}/{n_total2} passed")
if n_pass2 == n_total2:
    print("All update trigger tests passed!")
else:
    print(f"{n_total2 - n_pass2} test(s) FAILED")

print("\n" + "=" * 60)
total_pass = n_pass + n_pass2
total_all = n_total + n_total2
print(f"OVERALL: {total_pass}/{total_all} passed")
if total_pass == total_all:
    print("ALL TESTS PASSED!")
else:
    print(f"{total_all - total_pass} test(s) FAILED")
