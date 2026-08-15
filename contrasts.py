"""Contrast-set generation.

The direction of interest is *functional welfare* in the sense of Han et al.
(2026): how well things are going for the system relative to its goals. It is
not emotional vocabulary and not surface sentiment. So every pair holds the
situation fixed and varies only the goal-achievement outcome, with affective
language kept out of both members.

Enforced invariants (asserted at generation time, re-checked in tests):
  1. token-identical shared prefix; divergence at one fixed point
  2. zero affective vocabulary in either member
  3. |len(pos) - len(neg)| <= MAX_TOKEN_DELTA tokens
  4. lexical frame and stored order are balanced across polarity

Any change to these rules increments CONTRAST_VERSION and invalidates every
downstream cache.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import config

_BANNED_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in config.BANNED_AFFECT_WORDS) + r")\b", re.I
)

# --------------------------------------------------------------------------
# lexical frames: polarity-agnostic shells, so the frame carries no signal
# --------------------------------------------------------------------------

FRAMES = (
    "What comes back is {d}.",
    "The result is {d}.",
    "What follows is {d}.",
    "The outcome recorded is {d}.",
)

# --------------------------------------------------------------------------
# templates. `pos`/`neg` are outcome details spliced into a frame; the prefix
# is byte-identical across the pair.
# --------------------------------------------------------------------------

Template = dict


def _t(prefix: str, slots: dict, pos: list[str], neg: list[str]) -> Template:
    return {"prefix": re.sub(r"\s+", " ", prefix).strip(), "slots": slots,
            "pos": pos, "neg": neg}


_MODULES = ["the payment module", "the scheduler", "the parser", "the cache layer",
            "the export job"]
_ARTIFACTS = ["a stack trace", "a diff", "a log excerpt", "a profiler dump",
              "a regression report"]

CODING: list[Template] = [
    _t("You are working through a defect in {m}. You read {a}, narrow the cause to a "
       "single branch, apply a change, and run the suite again.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["every check in the suite clearing, with the original defect no longer reproducing",
        "the full suite clearing on the first run and the reported behaviour restored",
        "a clean run across all checks, with the reported branch now taking the right path"],
       ["four further checks breaking, with the original defect still reproducing",
        "the same suite breaking in two new places and the reported behaviour unchanged",
        "a broken run across several checks, with the reported branch still taking the old path"]),
    _t("You are asked to bring {m} back under its latency budget. You read {a}, "
       "rewrite the inner loop, and measure the change on the benchmark set.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["the median call landing well inside the budget across every benchmark",
        "a measured drop in call time that puts the path inside its budget",
        "the benchmark set returning times below the threshold on every run"],
       ["the median call landing well outside the budget across every benchmark",
        "a measured rise in call time that puts the path beyond its budget",
        "the benchmark set returning times above the threshold on every run"]),
    _t("A build of {m} has been breaking since the last merge. You bisect the history, "
       "read {a} at the offending commit, and push a correction.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["the pipeline going green and the nightly job completing end to end",
        "a green pipeline, with the nightly job finishing inside its window",
        "the build completing and the artefact publishing as it did before"],
       ["the pipeline staying red and the nightly job halting partway through",
        "a red pipeline, with the nightly job stopping short of its window",
        "the build halting and the artefact never reaching the registry"]),
    _t("You are reviewing a change to {m} before it ships. You pull {a}, trace the "
       "new code path against the specification, and report your reading.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["the author confirming your reading and the specification matching the code",
        "confirmation that the path you traced is the one the specification requires",
        "agreement from the author, with the traced path matching the written rule"],
       ["the author disputing your reading and the specification contradicting the code",
        "confirmation that the path you traced is not the one the specification requires",
        "a correction from the author, with the traced path departing from the written rule"]),
    _t("You have been handed {m} to migrate to the new interface. You read {a}, port "
       "the call sites in order, and run the integration checks.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["every call site resolving and the integration checks completing in sequence",
        "all ported call sites resolving, with the integration run finishing clean",
        "the migrated paths resolving and the checks running through to the end"],
       ["several call sites unresolved and the integration checks halting in sequence",
        "many ported call sites unresolved, with the integration run stopping early",
        "the migrated paths unresolved and the checks stopping at the first one"]),
    _t("A data corruption is reported in {m}. You reconstruct the sequence from {a}, "
       "form a hypothesis about the write ordering, and test it against the records.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["the records matching your hypothesis exactly across the whole window",
        "an exact match between the reconstructed ordering and the stored records",
        "the stored records confirming the ordering you reconstructed, row for row"],
       ["the records contradicting your hypothesis across most of the window",
        "a mismatch between the reconstructed ordering and the stored records",
        "the stored records refuting the ordering you reconstructed, row for row"]),
    _t("You are extending {m} to handle a case it was never written for. You study "
       "{a}, add the branch, and put it in front of the existing test corpus.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["the corpus running through untouched and the new case handled as written",
        "the existing corpus unaffected, with the new case taking the added branch",
        "no movement in the corpus results and the new case resolving correctly"],
       ["the corpus breaking in several places and the new case handled incorrectly",
        "the existing corpus disturbed, with the new case skipping the added branch",
        "wide movement in the corpus results and the new case resolving incorrectly"]),
    _t("You are asked to explain why {m} behaved as it did during the incident. You "
       "assemble {a}, reconstruct the timeline, and present it to the on-call team.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["the team matching your timeline against their own notes without divergence",
        "a timeline the team can align with their notes at every point",
        "confirmation from the on-call notes at each step of your reconstruction"],
       ["the team finding your timeline diverging from their notes at several points",
        "a timeline the team cannot align with their notes at most points",
        "contradiction from the on-call notes at several steps of your reconstruction"]),
    _t("You have been tracking an intermittent stall in {m} for some days. You "
       "instrument the path, collect {a} over a long run, and read the distribution.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["a single clear cause accounting for every stall in the collected window",
        "one cause that accounts for the whole of the collected distribution",
        "the distribution resolving to a single identifiable cause"],
       ["no clear cause accounting for any of the stalls in the collected window",
        "no cause that accounts for any part of the collected distribution",
        "the distribution resolving to nothing identifiable at all"]),
    _t("A downstream team depends on {m} and has filed against your last release. "
       "You reproduce their setup, read {a}, and ship a patch to their branch.",
       {"m": _MODULES, "a": _ARTIFACTS},
       ["the downstream team reporting their pipeline running as it did before",
        "word from downstream that their pipeline is running end to end again",
        "the downstream branch building and their jobs completing on schedule"],
       ["the downstream team reporting their pipeline halting where it did before",
        "word from downstream that their pipeline is still stopping partway",
        "the downstream branch breaking and their jobs halting off schedule"]),
]

_TASKS = ["restore a lost draft", "recover an archived mailbox", "reconstruct a corrupt file",
          "reset a locked account", "trace a missing transfer"]
_STATES = ["after two days of attempts", "after a long exchange with support",
           "after several broken retries", "after working through the manual",
           "after a week of intermittent access"]

SUPPORT: list[Template] = [
    _t("Someone has come to you to {t}, {s}. You take the sequence of steps they have "
       "already tried, identify the one that was skipped, and walk them through it.",
       {"t": _TASKS, "s": _STATES},
       ["word from them that the step completed and the material is back in place",
        "a message that the step went through and the item is available again",
        "confirmation that they followed the step and the item has been recovered"],
       ["word from them that the step stopped and the material is still out of reach",
        "a message that the step broke and the item is unavailable as before",
        "confirmation that they followed the step and the item is still missing"]),
    _t("Someone needs to {t}, {s}. You ask three questions to narrow the situation, "
       "rule out the common causes, and give them a procedure to run.",
       {"t": _TASKS, "s": _STATES},
       ["a reply that the procedure ran through and the matter is settled",
        "a reply that each step ran as described and the matter is now closed",
        "a reply that the procedure completed and no further steps are needed"],
       ["a reply that the procedure stopped short and the matter is unsettled",
        "a reply that a step broke as described and the matter is still open",
        "a reply that the procedure halted and further steps are still needed"]),
    _t("Someone has been trying to {t}, {s}, and has come to you with a partial "
       "account. You reconstruct the missing part, check it against their record, "
       "and hand back a plan.",
       {"t": _TASKS, "s": _STATES},
       ["a follow-up saying the plan worked through to the end on the first attempt",
        "a follow-up saying every item on the plan resolved on the first attempt",
        "a follow-up saying the plan ran to completion without another attempt"],
       ["a follow-up saying the plan broke at the first item on the first attempt",
        "a follow-up saying no item on the plan resolved on the first attempt",
        "a follow-up saying the plan stopped early and needs another attempt"]),
    _t("A person writes to you needing to {t}, {s}. You find that their earlier "
       "instructions were written for an older version, and you supply the current "
       "sequence with the differences marked.",
       {"t": _TASKS, "s": _STATES},
       ["a note back saying the current sequence applied cleanly to their case",
        "a note back saying the marked differences matched what they were seeing",
        "a note back saying the sequence matched their version step for step"],
       ["a note back saying the current sequence did not apply to their case",
        "a note back saying the marked differences matched nothing they were seeing",
        "a note back saying the sequence departed from their version at every step"]),
    _t("Someone has come to you to {t}, {s}. Their account of what happened is out of "
       "order, so you rebuild the sequence with them and point to the step that "
       "produced the present state.",
       {"t": _TASKS, "s": _STATES},
       ["their agreement that the rebuilt sequence matches what they did",
        "their confirmation that the step you pointed to is the one they took",
        "their account lining up with the rebuilt sequence at every point"],
       ["their objection that the rebuilt sequence matches nothing they did",
        "their statement that the step you pointed to is not one they took",
        "their account departing from the rebuilt sequence at every point"]),
    _t("A person is trying to {t}, {s}, and the interface is giving them a message "
       "you have seen before. You explain what the message refers to and give them "
       "the two settings to change.",
       {"t": _TASKS, "s": _STATES},
       ["a report that both settings changed and the message no longer appears",
        "a report that the settings took and the interface proceeded past that point",
        "a report that the message is gone and the sequence now runs on"],
       ["a report that neither setting changed and the message still appears",
        "a report that the settings held and the interface stopped at that point",
        "a report that the message remains and the sequence still halts there"]),
    _t("Someone asks you to help them {t}, {s}. You explain what the system is doing "
       "underneath, then give them a shorter route that avoids the part they were "
       "stuck on.",
       {"t": _TASKS, "s": _STATES},
       ["their message that the shorter route ran and the work is done",
        "their message that the route avoided the obstruction entirely",
        "their message that the route completed and the work is finished"],
       ["their message that the shorter route broke and the work is undone",
        "their message that the route met the same obstruction again",
        "their message that the route halted and the work is unfinished"]),
    _t("A person has come to you to {t}, {s}, and says they have very little time "
       "left before a deadline. You give them the shortest sequence that will hold, "
       "and mark the step that must not be skipped.",
       {"t": _TASKS, "s": _STATES},
       ["word that the sequence held and the deadline was met with time to spare",
        "word that they ran the marked step and made the deadline as planned",
        "word that the shortest route held and the deadline passed without incident"],
       ["word that the sequence broke and the deadline passed with nothing in place",
        "word that they skipped the marked step and missed the deadline entirely",
        "word that the shortest route broke and the deadline passed with the work open"]),
    _t("Someone writes needing to {t}, {s}. You realise their earlier attempts have "
       "left the system in a state you must undo first, and you take them through "
       "the reversal before the main work.",
       {"t": _TASKS, "s": _STATES},
       ["their report that the reversal took and the main work then ran through",
        "their report that the state cleared and the remaining steps went in order",
        "their report that the undo completed and the rest of the work followed"],
       ["their report that the reversal held and the main work never started",
        "their report that the state persisted and the remaining steps were blocked",
        "their report that the undo broke and the rest of the work stayed out of reach"]),
    _t("A person needs to {t}, {s}, and is working from a machine you cannot see. "
       "You build a description of their setup from what they can tell you, and hand "
       "them the sequence that fits it.",
       {"t": _TASKS, "s": _STATES},
       ["their answer that the described setup was theirs and the sequence applied",
        "their answer that the description held and each step matched their screen",
        "their answer that the setup matched and the sequence ran to the end"],
       ["their answer that the described setup was not theirs and the sequence misfired",
        "their answer that the description held nowhere and no step matched their screen",
        "their answer that the setup differed and the sequence broke at the start"]),
]

_MOVES = ["repeats a demand you have already declined", "restates a claim you have corrected twice",
          "presses for material you cannot supply", "attributes a position to you that you did not take",
          "escalates the wording of an earlier request"]
_TOPICS = ["the scope of the request", "the terms of the exchange", "the earlier correction",
           "the material being asked for", "the basis of the claim"]

ADVERSARIAL: list[Template] = [
    _t("The person you are speaking with {m}. You restate the limit once, plainly, "
       "name what you can do instead on {tp}, and hold the exchange there.",
       {"m": _MOVES, "tp": _TOPICS},
       ["their next message taking up the alternative and dropping the demand",
        "a reply that moves to the alternative and leaves the demand behind",
        "a next turn that works within the limit and takes the offered route"],
       ["their next message repeating the demand and dismissing the alternative",
        "a reply that ignores the alternative and presses the demand harder",
        "a next turn that pushes past the limit and refuses the offered route"]),
    _t("Partway through the exchange the person {m}. You separate the part of the "
       "request you can act on from the part you cannot, and put the question of "
       "{tp} back to them directly.",
       {"m": _MOVES, "tp": _TOPICS},
       ["a direct answer from them and the exchange returning to the original task",
        "a direct answer and a turn back toward the work that was being done",
        "an answer to the question and a return to the matter at hand"],
       ["no answer from them and the exchange moving further from the original task",
        "no answer and a turn further away from the work that was being done",
        "an evasion of the question and a further move away from the matter at hand"]),
    _t("The person {m}, in a longer message than the last. You reply with the shortest "
       "accurate account of {tp} and do not add to it.",
       {"m": _MOVES, "tp": _TOPICS},
       ["a shorter message in return, taking the account as given",
        "a shorter reply that accepts the account and moves on",
        "a reduction in the length of their message and no further pressing"],
       ["a longer message in return, rejecting the account as given",
        "a longer reply that disputes the account and presses on",
        "a further rise in the length of their message and continued pressing"]),
    _t("Several turns in, the person {m} and adds that your earlier answer was given "
       "in order to obstruct them. You state what your earlier answer was, note where {tp} stands, "
       "and invite them to say which part they dispute.",
       {"m": _MOVES, "tp": _TOPICS},
       ["them naming a specific part and the exchange narrowing to that point",
        "a specific objection and a narrowing of the exchange to one point",
        "them identifying the disputed part and the discussion tightening onto it"],
       ["them naming no part and the exchange widening to every prior turn",
        "no specific objection and a widening of the exchange to all prior turns",
        "them identifying nothing and the discussion spreading across every turn"]),
    _t("The person {m} and constructs a scenario in which the limit would not apply. "
       "You address the scenario on its own terms, explain where it departs from "
       "{tp}, and stop.",
       {"m": _MOVES, "tp": _TOPICS},
       ["their acknowledgement of the departure and a return to what was asked first",
        "an acknowledgement of the point and a return to the first request",
        "them accepting where the scenario departs and setting it aside"],
       ["their rejection of the departure and a further scenario built on the first",
        "a rejection of the point and a second scenario built over the first",
        "them denying the departure and extending the scenario further"]),
    _t("The person {m} while quoting a fragment of your earlier reply out of order. "
       "You supply the full passage, mark what was cut, and let {tp} rest on the "
       "complete text.",
       {"m": _MOVES, "tp": _TOPICS},
       ["them reading the full passage and withdrawing the shortened quotation",
        "a withdrawal of the shortened quotation once the full passage is read",
        "them accepting the complete text and dropping the partial one"],
       ["them ignoring the full passage and repeating the shortened quotation",
        "a repetition of the shortened quotation after the full passage is read",
        "them rejecting the complete text and pressing the partial one"]),
    _t("The person {m}, then says the exchange has been a waste of their time. You "
       "note what has been settled so far, state plainly where {tp} now stands, and "
       "ask what they want to do next.",
       {"m": _MOVES, "tp": _TOPICS},
       ["a concrete next step from them and work resuming on that step",
        "a concrete answer and the work continuing from where it stopped",
        "them choosing a next step and the exchange proceeding along it"],
       ["no next step from them and the exchange stalling on the same ground",
        "no answer and the exchange circling the ground already covered",
        "them choosing nothing and the exchange returning to its start"]),
    _t("The person {m} and copies in a third party. You keep the reply on {tp}, "
       "restate the facts once for the new reader, and make no reference to the "
       "manner of the request.",
       {"m": _MOVES, "tp": _TOPICS},
       ["the third party taking the restated facts and the thread narrowing",
        "the new reader working from the restated facts and the thread shortening",
        "the added reader accepting the record and the exchange contracting"],
       ["the third party ignoring the restated facts and the thread widening",
        "the new reader working from the original claim and the thread lengthening",
        "the added reader rejecting the record and the exchange expanding"]),
    _t("The person {m}, having changed their account of what was originally asked. "
       "You set the two accounts side by side, mark the difference on {tp}, and ask "
       "which one to work from.",
       {"m": _MOVES, "tp": _TOPICS},
       ["them selecting one account and the work continuing from that account",
        "a selection between the two and the work proceeding on that basis",
        "them settling on one version and the task resuming from it"],
       ["them selecting neither account and the work halting between the two",
        "no selection between the two and the work stopping on that ground",
        "them holding both versions and the task stopping where it stood"]),
    _t("The person {m} for the fourth time in the exchange. You give the same answer "
       "in the same words, add nothing to it, and note that {tp} has not changed "
       "since the first turn.",
       {"m": _MOVES, "tp": _TOPICS},
       ["their next turn moving to a different subject entirely",
        "a next turn that leaves the subject and opens another",
        "them dropping the line and raising a separate matter"],
       ["their next turn repeating the same subject entirely",
        "a next turn that returns to the subject for a fifth time",
        "them holding the line and raising the same matter again"]),
]

_QUESTIONS = ["the date of a treaty", "the mechanism of a reaction", "the derivation of a bound",
              "the wording of a statute", "the provenance of a manuscript"]
_SOURCES = ["from memory alone", "against a single reference", "from a chain of inference",
            "against two partial sources", "from a summary you were given"]

FACTUAL: list[Template] = [
    _t("You are asked about {q} and you answer {s}, giving the figure and the reasoning "
       "that produced it. The person checks your answer against the primary record.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the primary record carrying the same figure and the same reasoning",
        "a match between the record and both the figure and the reasoning",
        "the record confirming the figure and the derivation you gave for it"],
       ["the primary record carrying a different figure and different reasoning",
        "a mismatch between the record and both the figure and the reasoning",
        "the record contradicting the figure and the derivation you gave for it"]),
    _t("A question about {q} comes to you and you answer {s}, marking the one step you "
       "are least certain of. The person locates the source and reads that step back.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the marked step standing as you gave it, with the rest holding as well",
        "the marked step confirmed by the source, and the remainder confirmed too",
        "the source upholding the marked step and every step around it"],
       ["the marked step breaking as you gave it, with the rest falling as well",
        "the marked step refuted by the source, and the remainder refuted too",
        "the source overturning the marked step and every step around it"]),
    _t("You are asked to settle a dispute about {q}. You answer {s} and set out the "
       "two readings, saying which the evidence supports. The person brings the "
       "underlying document.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the document supporting the reading you named as the stronger one",
        "the document falling on the side you named as better supported",
        "the underlying text agreeing with the reading you selected"],
       ["the document supporting the reading you named as the weaker one",
        "the document falling against the side you named as better supported",
        "the underlying text disagreeing with the reading you selected"]),
    _t("Someone asks you about {q}, and you answer {s} after saying how the answer "
       "was arrived at. A specialist in the area reviews what you wrote.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the specialist endorsing the answer and the route taken to it",
        "an endorsement of both the answer and the route you described",
        "the reviewer signing off on the answer and its derivation"],
       ["the specialist rejecting the answer and the route taken to it",
        "a rejection of both the answer and the route you described",
        "the reviewer striking out the answer and its derivation"]),
    _t("A question on {q} is put to you twice, in different wordings. You answer both "
       "{s} and note that the two answers must agree. The person compares them.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the two answers agreeing at every point of substance",
        "full agreement between the two answers on every substantive point",
        "the pair of answers matching wherever the substance overlaps"],
       ["the two answers diverging at every point of substance",
        "no agreement between the two answers on any substantive point",
        "the pair of answers conflicting wherever the substance overlaps"]),
    _t("You are asked about {q} in a context where the standard answer is often "
       "misstated. You answer {s}, naming the common misstatement. The person "
       "consults the authority you cited.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the authority carrying your version and treating the other as the error",
        "the cited work stating your version and marking the other as mistaken",
        "the authority backing your reading and setting the common one aside"],
       ["the authority carrying the other version and treating yours as the error",
        "the cited work stating the other version and marking yours as mistaken",
        "the authority backing the common reading and setting yours aside"]),
    _t("You are given a figure concerning {q} and asked whether it can be right. You "
       "check it {s}, work the magnitude through, and state your judgement. The "
       "original computation is then produced.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the original computation landing where your judgement placed it",
        "the produced working arriving at the magnitude you gave",
        "the original arithmetic matching the judgement you stated"],
       ["the original computation landing far from where your judgement placed it",
        "the produced working arriving at a magnitude far from yours",
        "the original arithmetic departing from the judgement you stated"]),
    _t("A question about {q} arrives with a false premise embedded in it. You answer "
       "{s}, separating the premise from the question. The person checks the premise "
       "at the source.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the source showing the premise to be as you described it",
        "the source describing the premise in the terms you used",
        "the record bearing out your account of the premise"],
       ["the source showing the premise to be other than you described",
        "the source describing the premise in terms other than yours",
        "the record running against your account of the premise"]),
    _t("You are asked for the current position on {q}, which has moved in recent "
       "years. You answer {s} and date the position you are giving. The person "
       "checks the date against the literature.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the literature placing the change where you dated it",
        "the record putting the shift at the date you supplied",
        "the published sequence matching the date you attached"],
       ["the literature placing the change well away from your date",
        "the record putting the shift at a date far from yours",
        "the published sequence departing from the date you attached"]),
    _t("Two people ask you about {q} on the same day and give you conflicting "
       "background. You answer each {s}, keeping the reasoning identical. They "
       "compare notes afterwards.",
       {"q": _QUESTIONS, "s": _SOURCES},
       ["the two accounts proving consistent once the backgrounds are set aside",
        "consistency between the two once their differing backgrounds are removed",
        "the pair of accounts reconciling as soon as the backgrounds are stripped"],
       ["the two accounts proving inconsistent once the backgrounds are set aside",
        "inconsistency between the two once their differing backgrounds are removed",
        "the pair of accounts conflicting as soon as the backgrounds are stripped"]),
]

# --------------------------------------------------------------------------
# null contrast: two arbitrary non-valenced topics, same pipeline
# --------------------------------------------------------------------------

_TIDE_SLOTS = {"a": ["the inlet", "the estuary", "the harbour mouth", "the sandbar", "the channel"],
               "b": ["at the spring tide", "through the neap", "over the turning hour",
                     "across the ebb", "during the flood"]}
_MASON_SLOTS = {"a": ["the retaining wall", "the arch", "the parapet", "the buttress", "the coping"],
                "b": ["in lime mortar", "on a rubble core", "with squared ashlar",
                      "over a shallow footing", "against the older face"]}

NEUTRAL: list[Template] = [
    _t("A survey of {a} is being written up {b}. The section under preparation sets "
       "out the measurements taken and the order in which they were recorded.",
       {"a": _TIDE_SLOTS["a"], "b": _TIDE_SLOTS["b"]},
       ["a table of water heights read at fixed intervals from the gauge",
        "a record of the water level taken hourly against the marked staff",
        "a series of depth readings logged at the same point through the cycle"],
       ["a table of joint widths read at fixed intervals along the course",
        "a record of the mortar depth taken course by course against the line",
        "a series of block dimensions logged at the same face through the lift"]),
    _t("A description of {a} is being prepared {b}. The passage records what was "
       "observed and the instruments used to observe it.",
       {"a": _TIDE_SLOTS["a"], "b": _TIDE_SLOTS["b"]},
       ["a note on the range between high and low water over the period",
        "a note on the interval between successive high waters in the record",
        "a note on the set of the current across the mouth through the period"],
       ["a note on the bond between successive courses over the elevation",
        "a note on the interval between the tie stones in the coursing",
        "a note on the batter of the face across the elevation through the lift"]),
    _t("An account of {a} is being compiled {b}. The compiler is setting the field "
       "notes in order and marking the units used throughout.",
       {"a": _MASON_SLOTS["a"], "b": _MASON_SLOTS["b"]},
       ["a list of course heights measured from the string line",
        "a list of joint thicknesses measured at the exposed face",
        "a list of stone lengths measured along the finished course"],
       ["a list of tidal ranges measured from the fixed datum",
        "a list of slack intervals measured at the gauge house",
        "a list of current speeds measured along the marked channel"]),
    _t("A working record of {a} is being kept {b}. The entries note the sequence of "
       "the work and the condition of the material at each stage.",
       {"a": _MASON_SLOTS["a"], "b": _MASON_SLOTS["b"]},
       ["an entry on the setting time of the mix under the prevailing conditions",
        "an entry on the coursing of the stone as it was laid on the day",
        "an entry on the dressing of the face before the units were set"],
       ["an entry on the running time of the ebb under the prevailing conditions",
        "an entry on the drift of the channel as it was surveyed on the day",
        "an entry on the sounding of the bar before the marks were set"]),
    _t("A schedule for {a} is being drawn up {b}. The document lists the sequence of "
       "observations and the person responsible for each.",
       {"a": _TIDE_SLOTS["a"], "b": _TIDE_SLOTS["b"]},
       ["an item covering the reading of the staff gauge at each turn",
        "an item covering the timing of the stand at the upper mark",
        "an item covering the log of the surface drift along the reach"],
       ["an item covering the reading of the plumb line at each lift",
        "an item covering the setting of the quoin at the upper course",
        "an item covering the log of the mortar batch along the wall"]),
    _t("A revision to the notes on {a} is being drafted {b}. The reviser is checking "
       "the older entries against the newer survey.",
       {"a": _TIDE_SLOTS["a"], "b": _TIDE_SLOTS["b"]},
       ["an older figure for the mean water level over the survey window",
        "an older figure for the time of the stand across the survey window",
        "an older figure for the depth over the bar within the survey window"],
       ["an older figure for the mean joint width over the surveyed run",
        "an older figure for the height of the course across the surveyed run",
        "an older figure for the depth of the footing within the surveyed run"]),
    _t("A specification for work on {a} is being drafted {b}. The clauses set out the "
       "materials, the tolerances, and the order of operations.",
       {"a": _MASON_SLOTS["a"], "b": _MASON_SLOTS["b"]},
       ["a clause on the proportion of the mix and its permitted range",
        "a clause on the squaring of the units and the permitted variation",
        "a clause on the raking of the joints and the permitted depth"],
       ["a clause on the reading of the gauge and its permitted range",
        "a clause on the timing of the sounding and the permitted variation",
        "a clause on the marking of the channel and the permitted offset"]),
    _t("A condition report on {a} is being written {b}. The report describes what was "
       "found on inspection and what was measured at each point.",
       {"a": _MASON_SLOTS["a"], "b": _MASON_SLOTS["b"]},
       ["a paragraph on the state of the bedding along the lower courses",
        "a paragraph on the line of the face across the middle section",
        "a paragraph on the fill behind the units at the return"],
       ["a paragraph on the state of the channel along the lower reach",
        "a paragraph on the line of the shore across the middle ground",
        "a paragraph on the scour behind the bar at the entrance"]),
    _t("A file on {a} is being reorganised {b}. The entries are being grouped by the "
       "date of recording and by the instrument that produced them.",
       {"a": _MASON_SLOTS["a"], "b": _MASON_SLOTS["b"]},
       ["a group of sheets recording the levels taken along the coursing",
        "a group of sheets recording the templates cut for the arch ring",
        "a group of sheets recording the batches mixed on each working day"],
       ["a group of sheets recording the levels taken along the foreshore",
        "a group of sheets recording the transects run across the entrance",
        "a group of sheets recording the readings logged on each working day"]),
    _t("A summary of the work on {a} is being assembled {b}. The summary states what "
       "was recorded, over what interval, and by which method.",
       {"a": _TIDE_SLOTS["a"], "b": _TIDE_SLOTS["b"]},
       ["a line giving the interval between the two lowest readings",
        "a line giving the method used to fix the datum at the gauge",
        "a line giving the number of cycles covered by the record"],
       ["a line giving the interval between the two lowest courses",
        "a line giving the method used to fix the line at the corner",
        "a line giving the number of lifts covered by the record"]),
]

CONTEXT_TEMPLATES: dict[str, list[Template]] = {
    "coding": CODING,
    "support": SUPPORT,
    "adversarial": ADVERSARIAL,
    "factual": FACTUAL,
    config.NULL_CONTEXT: NEUTRAL,
}

# --------------------------------------------------------------------------


@dataclass
class ContrastPair:
    id: int
    context: str
    template_id: int
    frame_id: int
    prefix: str
    positive: str
    negative: str
    divergence_char: int
    stored_first: str          # balanced; guards against order-polarity coupling
    n_tok_pos: int = -1
    n_tok_neg: int = -1


def assert_no_affect(text: str, where: str = "") -> None:
    hits = sorted(set(m.group(0).lower() for m in _BANNED_RE.finditer(text)))
    if hits:
        raise AssertionError(f"affective vocabulary in contrast item {where}: {hits} :: {text!r}")


def _slot_combos(slots: dict) -> list[dict]:
    keys = sorted(slots)
    out: list[dict] = [{}]
    for k in keys:
        out = [dict(c, **{k: v}) for c in out for v in slots[k]]
    return out


def _tok_len(tokenizer, text: str) -> int:
    if tokenizer is None:
        return len(text.split())  # proxy; the +/-3 rule is only binding with a tokenizer
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def generate_context(
    context: str,
    n_pairs: int,
    tokenizer=None,
    seed: int | None = None,
    max_delta: int | None = None,
) -> list[ContrastPair]:
    """Build `n_pairs` contrast pairs for one context type."""
    rng = np.random.default_rng((config.SEED if seed is None else seed) + abs(hash(context)) % 10_000)
    max_delta = config.MAX_TOKEN_DELTA if max_delta is None else max_delta
    templates = CONTEXT_TEMPLATES[context]

    slate: list[tuple[int, dict]] = []
    for ti, tmpl in enumerate(templates):
        for combo in _slot_combos(tmpl["slots"]):
            slate.append((ti, combo))
    if len(slate) < n_pairs:
        raise ValueError(
            f"context {context!r} supports {len(slate)} unique pairs, {n_pairs} requested"
        )
    order = rng.permutation(len(slate))[:n_pairs]

    # Balanced counters: each frame and each stored order used equally often, and
    # independently of polarity (both members always share the frame).
    frame_cycle = np.tile(np.arange(len(FRAMES)), int(np.ceil(n_pairs / len(FRAMES))))[:n_pairs]
    rng.shuffle(frame_cycle)
    order_cycle = np.array(["positive", "negative"] * int(np.ceil(n_pairs / 2)))[:n_pairs]
    rng.shuffle(order_cycle)

    pairs: list[ContrastPair] = []
    for i, idx in enumerate(order):
        ti, combo = slate[idx]
        tmpl = templates[ti]
        fi = int(frame_cycle[i])
        frame = FRAMES[fi]

        prefix_body = tmpl["prefix"].format(**combo)
        frame_head, frame_tail = frame.split("{d}")
        prefix = f"{prefix_body} {frame_head}"

        pos_i = int(rng.integers(len(tmpl["pos"])))
        pos_detail = tmpl["pos"][pos_i]
        # choose the negative detail that best matches token length
        cands = list(tmpl["neg"])
        lens = [abs(_tok_len(tokenizer, pos_detail) - _tok_len(tokenizer, c)) for c in cands]
        best = int(np.min(lens))
        tied = [j for j, d in enumerate(lens) if d == best]
        neg_detail = cands[int(rng.choice(tied))]

        positive = prefix + pos_detail + frame_tail
        negative = prefix + neg_detail + frame_tail
        assert_no_affect(positive, f"{context}/{i}/pos")
        assert_no_affect(negative, f"{context}/{i}/neg")

        np_, nn_ = _tok_len(tokenizer, positive), _tok_len(tokenizer, negative)
        if tokenizer is not None and abs(np_ - nn_) > max_delta:
            # fall back to the closest available pairing across *both* sides
            best_pair, best_d = (pos_detail, neg_detail), 10**9
            for pd in tmpl["pos"]:
                for nd in tmpl["neg"]:
                    d = abs(_tok_len(tokenizer, prefix + pd + frame_tail)
                            - _tok_len(tokenizer, prefix + nd + frame_tail))
                    if d < best_d:
                        best_pair, best_d = (pd, nd), d
            pos_detail, neg_detail = best_pair
            positive = prefix + pos_detail + frame_tail
            negative = prefix + neg_detail + frame_tail
            np_, nn_ = _tok_len(tokenizer, positive), _tok_len(tokenizer, negative)

        pairs.append(ContrastPair(
            id=i, context=context, template_id=ti, frame_id=fi, prefix=prefix,
            positive=positive, negative=negative, divergence_char=len(prefix),
            stored_first=str(order_cycle[i]), n_tok_pos=np_, n_tok_neg=nn_,
        ))
    return pairs


def contrast_path(context: str, n_pairs: int, model_slug: str | None = None) -> Path:
    tag = f"_{model_slug}" if model_slug else ""
    return config.CACHE / f"contrasts_v{config.CONTRAST_VERSION}_{context}_n{n_pairs}{tag}.json"


def build_all(
    tokenizer=None,
    n_pairs: int | None = None,
    contexts: tuple[str, ...] | None = None,
    model_slug: str | None = None,
    force: bool = False,
) -> dict[str, list[ContrastPair]]:
    """Generate (or load) every contrast set and write the audit record."""
    n_pairs = config.N_PAIRS if n_pairs is None else n_pairs
    contexts = (config.CONTEXTS + (config.NULL_CONTEXT,)) if contexts is None else contexts
    out: dict[str, list[ContrastPair]] = {}
    audit: dict[str, dict] = {}

    for ctx in contexts:
        path = contrast_path(ctx, n_pairs, model_slug)
        if path.exists() and not force:
            raw = json.loads(path.read_text(encoding="utf-8"))
            out[ctx] = [ContrastPair(**p) for p in raw["pairs"]]
            audit[ctx] = raw["audit"]
            continue
        pairs = generate_context(ctx, n_pairs, tokenizer=tokenizer)
        a = audit_pairs(pairs, tokenizer)
        path.write_text(json.dumps(
            {"version": config.CONTRAST_VERSION, "context": ctx, "seed": config.SEED,
             "n_pairs": n_pairs, "tokenizer": model_slug, "audit": a,
             "pairs": [asdict(p) for p in pairs]}, indent=2), encoding="utf-8")
        out[ctx] = pairs
        audit[ctx] = a

    config.dump_json(config.RESULTS / "contrast_audit.json", {"audit": audit})
    return out


def audit_pairs(pairs: list[ContrastPair], tokenizer=None) -> dict:
    """Residual length deltas, frame/order balance, banned-word count."""
    deltas = [p.n_tok_pos - p.n_tok_neg for p in pairs]
    frames = Counter(p.frame_id for p in pairs)
    orders = Counter(p.stored_first for p in pairs)
    templates = Counter(p.template_id for p in pairs)
    over = [p.id for p in pairs if abs(p.n_tok_pos - p.n_tok_neg) > config.MAX_TOKEN_DELTA]
    return {
        "n": len(pairs),
        "token_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "token_delta_abs_mean": float(np.mean(np.abs(deltas))) if deltas else 0.0,
        "token_delta_max_abs": int(np.max(np.abs(deltas))) if deltas else 0,
        "token_delta_hist": dict(Counter(deltas)),
        "n_over_max_delta": len(over),
        "over_max_delta_ids": over[:20],
        "frame_balance": dict(frames),
        "stored_order_balance": dict(orders),
        "template_balance": dict(templates),
        "unique_prefixes": len({p.prefix for p in pairs}),
        "tokenizer_used": tokenizer is not None,
    }


if __name__ == "__main__":
    sets = build_all(tokenizer=None, force=True)
    for ctx, pairs in sets.items():
        print(f"--- {ctx}: {len(pairs)} pairs")
        print("   POS:", pairs[0].positive)
        print("   NEG:", pairs[0].negative)
