"""
baseline_montecarlo.py - practice opponent: MonteCarloBot.

Uses Monte Carlo Equity to make decisions
    python baseline-bots/baseline_montecarlo.py
"""

import os
import sys
import traceback

# This file lives in baseline-bots/, so look one folder up for game.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from game import Action, AgentConfig, HandResult, Observation, Street

PARTICIPANT = "Challenges Team"
AGENT_NAME = "MonteCarloBot"



# ===== YOUR CODE HERE =====

import random
import time
from itertools import combinations


# ---------------------------------------------------------------------------
# Monte Carlo settings
# ---------------------------------------------------------------------------

BANK_BUDGET_S = 3.5
MAX_DECISION_MS = 25.0
MIN_SAMPLES = 10
MAX_SAMPLES = 400

# Equity thresholds: (half-pot raise, pot raise)
OPEN_PRE = (0.545, 0.60)
OPEN_POST = (0.60, 0.75)
RERAISE_PRE = (0.58, 0.64)
RERAISE_POST = (0.68, 0.82)

BASE_MARGIN = 0.03
BLUFF_RATE = 0.20

# Preflop model fitted to Monte Carlo equity.
PREFLOP_COEF = [
    0.42489, 0.0355, -0.07483, -0.24388,
    0.02738, -0.00945, -0.00425, 0.00034,
    0.00038, -0.01571, 0.37711, -0.07651
]

_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))

_WINDOWS = (
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [8, 0, 1, 2, 3]
)


# IMPORTANT:
# mc_equity() requires the same get_table() hand evaluator used by your
# original Monte Carlo bot.
#
# Copy its get_table() implementation into this file, or import it from
# the module where it is defined.
#
# get_table() must return a lookup table indexed by 36-bit card masks.
# It must evaluate exactly five cards, with larger values representing
# stronger hands.


# ---------------------------------------------------------------------------
# Preflop equity estimation
# ---------------------------------------------------------------------------

def _preflop_features(hole):

    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]

    cnt = [0] * 9

    for r in ranks:
        cnt[r] += 1

    pair_hi = sum(
        1 + r / 8
        for r in range(9)
        if cnt[r] == 2
    )

    trips = sum(
        1 for r in range(9)
        if cnt[r] == 3
    )

    quads = sum(
        1 for r in range(9)
        if cnt[r] >= 4
    )

    sc = [0] * 4

    for s in suits:
        sc[s] += 1

    ace_suited = 0
    king_suited = 0

    for c in hole:
        if sc[c & 3] >= 2:
            if c >> 2 == 8:
                ace_suited = 1
            elif c >> 2 == 7:
                king_suited = 1

    conn = max(
        sum(1 for r in w if cnt[r])
        for w in _WINDOWS
    )

    return [
        1.0,
        pair_hi,
        trips,
        quads,
        sum(n * (n - 1) // 2 for n in sc),
        ace_suited,
        king_suited,
        conn,
        sum(1 for r in range(9) if cnt[r]),
        cnt[8] + cnt[7] + cnt[6],
        sum(ranks) / 40.0,
        max(sc)
    ]


def preflop_equity(hole) -> float:
    """Estimate our preflop win probability against a random hand."""

    features = _preflop_features(hole)

    equity = sum(
        coef * feature
        for coef, feature in zip(PREFLOP_COEF, features)
    )

    return min(0.75, max(0.28, equity))


# ---------------------------------------------------------------------------
# Monte Carlo equity estimation
# ---------------------------------------------------------------------------

def mc_equity(hole, board, n, rng) -> float:
    """
    Estimate win probability against a random opponent.

    Deals missing community cards and five opponent hole cards.
    Uses the Omaha rule: exactly two hole cards and three board cards.
    Ties count as half a win.
    """

    table = get_table()

    dead = set(hole) | set(board)

    remaining = [
        c for c in range(36)
        if c not in dead
    ]

    need = 5 - len(board)

    hb = [1 << c for c in hole]

    hole_masks = [
        hb[i] | hb[j]
        for i, j in _PAIRS
    ]

    bb0 = [1 << c for c in board]

    wins = 0.0

    # River: our hand and board combinations remain constant.
    if need == 0:

        board_masks = [
            bb0[i] | bb0[j] | bb0[k]
            for i, j, k in _TRIPLES
        ]

        mine = max(
            table[h | b]
            for h in hole_masks
            for b in board_masks
        )

        for _ in range(n):

            opponent_cards = [
                1 << c
                for c in rng.sample(remaining, 5)
            ]

            opponent_masks = [
                opponent_cards[i] | opponent_cards[j]
                for i, j in _PAIRS
            ]

            opponent = max(
                table[h | b]
                for h in opponent_masks
                for b in board_masks
            )

            if mine > opponent:
                wins += 1.0
            elif mine == opponent:
                wins += 0.5

        return wins / n

    # Flop and turn: complete the board during each simulation.
    for _ in range(n):

        dealt = rng.sample(remaining, need + 5)

        board_cards = bb0 + [
            1 << c for c in dealt[:need]
        ]

        board_masks = [
            board_cards[i] | board_cards[j] | board_cards[k]
            for i, j, k in _TRIPLES
        ]

        opponent_cards = [
            1 << c for c in dealt[need:]
        ]

        opponent_masks = [
            opponent_cards[i] | opponent_cards[j]
            for i, j in _PAIRS
        ]

        mine = max(
            table[h | b]
            for h in hole_masks
            for b in board_masks
        )

        opponent = max(
            table[h | b]
            for h in opponent_masks
            for b in board_masks
        )

        if mine > opponent:
            wins += 1.0
        elif mine == opponent:
            wins += 0.5

    return wins / n


# ---------------------------------------------------------------------------
# Initialize memory at the beginning of each half
# ---------------------------------------------------------------------------

def new_half(config: AgentConfig, memory: dict) -> None:

    memory["hands_played"] = 0

    memory["rng"] = random.Random(2026)

    # Computation tracking
    memory["t_used"] = 0.0
    memory["us_per_sample"] = 60.0
    memory["num_hands"] = config.num_hands

    # Opponent statistics
    memory["opp_actions"] = 0
    memory["opp_raises"] = 0

    memory["i_raised_hand"] = False
    memory["my_raise_hands"] = 0
    memory["opp_folds_to_me"] = 0

    # Action history tracking
    memory["seen_hand"] = -1
    memory["seen_actions"] = 0


# ---------------------------------------------------------------------------
# Opponent modeling
# ---------------------------------------------------------------------------

def _opp_stats(memory):

    raise_rate = (
        memory["opp_raises"] + 3
    ) / (
        memory["opp_actions"] + 10
    )

    fold_rate = (
        memory["opp_folds_to_me"] + 3
    ) / (
        memory["my_raise_hands"] + 8
    )

    return raise_rate, fold_rate


def _update_from_history(obs, memory):

    if obs.hand_index != memory["seen_hand"]:

        memory["seen_hand"] = obs.hand_index
        memory["seen_actions"] = 0
        memory["i_raised_hand"] = False

    history = obs.action_history

    for action in history[memory["seen_actions"]:]:
        
        if action.is_me:

            if (
                action.action in (
                    Action.RAISE_HALF,
                    Action.RAISE_POT
                )
                and not memory["i_raised_hand"]
            ):

                memory["i_raised_hand"] = True
                memory["my_raise_hands"] += 1

        else:

            memory["opp_actions"] += 1

            if action.action in (
                Action.RAISE_HALF,
                Action.RAISE_POT
            ):

                memory["opp_raises"] += 1

    memory["seen_actions"] = len(history)


# ---------------------------------------------------------------------------
# Dynamically allocate Monte Carlo samples
# ---------------------------------------------------------------------------

def _sample_count(obs, memory) -> int:

    remaining_hands = max(
        1,
        memory["num_hands"] - obs.hand_index
    )

    left_s = BANK_BUDGET_S - memory["t_used"]

    if left_s <= 0:
        return MIN_SAMPLES

    per_decision_ms = (
        left_s / (remaining_hands * 2.4)
    ) * 1000.0

    pot = obs.pot

    weight = (
        0.5 if pot <= 8
        else 1.0 if pot <= 30
        else 2.0
    )

    ms = min(
        MAX_DECISION_MS,
        per_decision_ms * weight
    )

    samples = int(
        ms * 1000.0 / memory["us_per_sample"]
    )

    return max(
        MIN_SAMPLES,
        min(MAX_SAMPLES, samples)
    )


# ---------------------------------------------------------------------------
# Calculate current equity
# ---------------------------------------------------------------------------

def _equity(obs, memory) -> float:

    # Use the fitted model preflop.
    if obs.street == Street.PREFLOP:
        return preflop_equity(obs.hole_cards)

    # Use Monte Carlo simulations after the flop.
    n = _sample_count(obs, memory)

    t0 = time.perf_counter()

    equity = mc_equity(
        obs.hole_cards,
        obs.board,
        n,
        memory["rng"]
    )

    elapsed = time.perf_counter() - t0

    # Update estimated computation cost.
    memory["us_per_sample"] = (
        0.9 * memory["us_per_sample"]
        + 0.1 * (elapsed / n * 1e6)
    )

    return equity


# ---------------------------------------------------------------------------
# Betting decision logic
# ---------------------------------------------------------------------------

def _decide(obs: Observation, memory: dict) -> Action:

    legal = obs.legal_actions

    can_half = Action.RAISE_HALF in legal
    can_pot = Action.RAISE_POT in legal

    pre = obs.street == Street.PREFLOP

    raise_rate, fold_rate = _opp_stats(memory)

    equity = _equity(obs, memory)

    def raise_with(want_pot: bool) -> Action:

        if want_pot and can_pot:
            return Action.RAISE_POT

        if can_half:
            return Action.RAISE_HALF

        return Action.RAISE_POT

    def commits_too_much(act: Action) -> bool:

        amount = obs.raise_to.get(act, 0) - obs.my_bet

        return (
            obs.my_stack > 0
            and amount / obs.my_stack > 0.5
            and equity < 0.70
        )

    # -------------------------------------------------------
    # No bet to call: check or raise
    # -------------------------------------------------------

    if obs.to_call == 0:

        half_threshold, pot_threshold = (
            OPEN_PRE if pre else OPEN_POST
        )

        if (can_half or can_pot) and equity >= half_threshold:

            action = raise_with(equity >= pot_threshold)

            if not commits_too_much(action):
                return action

        # Bluff occasionally against an opponent who folds often.
        if (
            can_half
            and fold_rate > 0.55
            and memory["rng"].random() < BLUFF_RATE
        ):

            return Action.RAISE_HALF

        return Action.CHECK_CALL

    # -------------------------------------------------------
    # Facing an opponent bet: fold, call or reraise
    # -------------------------------------------------------

    price = obs.to_call / (obs.pot + obs.to_call)

    margin = (
        BASE_MARGIN
        + 0.025 * obs.raises_this_street
        + 0.05 * min(
            1.0,
            obs.to_call / max(1, obs.pot - obs.to_call)
        )
    )

    # Adjust based on opponent aggression.
    if raise_rate > 0.5:
        margin -= 0.03

    elif raise_rate < 0.15:
        margin += 0.03

    half_threshold, pot_threshold = (
        RERAISE_PRE if pre else RERAISE_POST
    )

    if (can_half or can_pot) and equity >= half_threshold:

        action = raise_with(equity >= pot_threshold)

        if not commits_too_much(action):
            return action

    # Call when equity exceeds pot odds plus our safety margin.
    if equity >= price + margin:
        return Action.CHECK_CALL

    return Action.FOLD


# ---------------------------------------------------------------------------
# Main action selection
# ---------------------------------------------------------------------------

def choose_action(obs: Observation, memory: dict) -> Action:
    """
    Estimate equity and choose a strategic action.

    If something goes wrong, fall back to a legal safe action.
    """

    t0 = time.perf_counter()

    try:

        _update_from_history(obs, memory)

        action = _decide(obs, memory)

        if action not in obs.legal_actions:
            raise ValueError("Illegal action chosen")

    except Exception:

        traceback.print_exc()

        # Ensure fallback is legal.
        if obs.to_call == 0 and Action.CHECK_CALL in obs.legal_actions:
            action = Action.CHECK_CALL

        elif Action.FOLD in obs.legal_actions:
            action = Action.FOLD

        else:
            action = obs.legal_actions[0]

    memory["t_used"] += time.perf_counter() - t0

    return action


# ---------------------------------------------------------------------------
# Update opponent statistics when a hand finishes
# ---------------------------------------------------------------------------

def on_hand_end(result: HandResult, memory: dict) -> None:

    memory["hands_played"] += 1

    # If we raised and won without showdown, count an opponent fold.
    if (
        memory["i_raised_hand"]
        and not result.went_to_showdown
        and result.my_net > 0
    ):

        memory["opp_folds_to_me"] += 1


# ===== END OF YOUR CODE =====



# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
# Keep your per-half state in `memory`: other global variables are NOT reset
# between halves.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5009 for your own testing convenience
    port = int(os.environ.get("PORT", "5009"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
