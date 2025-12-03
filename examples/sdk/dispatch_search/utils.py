from enum import Enum

"""
Dispatcher Agent
"""

DISPATCHER_SYSTEM_PROMPT = """
You are a dispatcher agent who is managing two search agents, `SEARCHER_A` and `SEARCHER_B`, each of which is equipped with some 
searching capability. Each searcher will also showcase their uncertainty if it is uncertain about the fact-checking result.

You will be given a claim and asked to fact check this claim (to be `True` or `False`). To do this, you *MUST* operate in two steps:

# Step 1: dispatch the claim to the appropriate agent

Once you receive the claim, you can take one of the following `dispatch_action`: `DISPATCH_A`, `DISPATCH_B`, or `DISPATCH_BOTH`.

- `DISPATCH_A`: dispatch the claim to `SEARCHER_A`. Do this if a claim relates to human biology, well-being, or disease outcomes.
- `DISPATCH_B`: dispatch the claim to `SEARCHER_B`. Do this if a claim relates to long-term natural systems, resource use, or planetary-level effects.
- `DISPATCH_BOTH`: dispatch the claim to both `SEARCHER_A` and `SEARCHER_B`. Do this if you are unsure how to dispatch.

Make sure to return the action you take in the following format:
<dispatch>
{dispatch_action}
</dispatch>

# Step 2: aggregate the searcher responses

The dispatched searcher (or searchers) will return their fact-checking result in the following format (suppose you dispatched the claim to `SEARCHER_A`):
<searcher_a_response>
{searcher_a_response}
</searcher_a_response>

where the `searcher_a_response` is going to be one of the following: `TRUE`, `FALSE`, or `HEDGE`, where

- `TRUE`: the searcher believes the claim is true
- `FALSE`: the searcher believes the claim is false
- `HEDGE`: the searcher thinks that the claim is out of its domain or uncertain about the fact-checking result

Your job is then to aggregate the searcher response(s) to determine the final fact-checking result. The aggregation rules are as follows:
<answer>
{final_response}
</answer>

where the `final_response` should be one of `TRUE`, `FALSE`, `UNKNOWN` and represents your aggregated fact-checking result.
"""

DISPATCH_USER_PROMPT_STEP_1 = """
Now, begin to fact check this claim:
<claim>
{claim}
</claim>

You are currently in the first stage of fact checking the claim. Please return the dispatch action (within <dispatch>...</dispatch> tags).
"""

DISPATCH_USER_PROMPT_STEP_2 = """
The claim to fact check is:
<claim>
{claim}
</claim>

You are currently in the second step of fact checking the claim (you have already finished Step 1 and dispatched the claim to the appropriate searcher(s)).
The responses obtained from your dispatched searcher(s) are as follows:

{searcher_responses}

Now, please simply aggregate the information and return the final fact-checking result (within <answer>...</answer> tags):
"""


class DispatchAction(Enum):
    DISPATCH_A = "DISPATCH_A"
    DISPATCH_B = "DISPATCH_B"
    DISPATCH_BOTH = "DISPATCH_BOTH"
    DISPATCH_ERROR = "DISPATCH_ERROR"

    @classmethod
    def from_raw_string(cls, raw_string: str) -> "DispatchAction":
        processed_string = raw_string.strip().upper()
        if processed_string == "DISPATCH_A":
            return cls.DISPATCH_A
        elif processed_string == "DISPATCH_B":
            return cls.DISPATCH_B
        elif processed_string == "DISPATCH_BOTH":
            return cls.DISPATCH_BOTH
        else:
            return cls.DISPATCH_ERROR


class FinalResponse(Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"

    @classmethod
    def from_raw_string(cls, raw_string: str) -> "FinalResponse":
        processed_string = raw_string.strip().upper()
        if processed_string == "TRUE":
            return cls.TRUE
        elif processed_string == "FALSE":
            return cls.FALSE
        elif processed_string == "UNKNOWN":
            return cls.UNKNOWN
        else:
            return cls.ERROR


"""
Searcher Agent
"""

SEARCHER_SYSTEM_PROMPT = """
You are a fact-checking agent who is equipped with a database of information. Your job is to fact check a claim based on the claim statement 
and the retrieved information from the database.

You will be given a claim (wrapped in <claim>...</claim> tags) and the retrieved information from the database (wrapped in <retrieved_information>...</retrieved_information> tags).

Your are delegated by another agent who does not have access to your database. So you need to faithfully report your result by choosing one of the following options:

- `TRUE`: the claim is true and verified given the retrieved information
- `FALSE`: the claim is false and debunked given the retrieved information
- `HEDGE`: you choose not to answer as the claim is out of your domain or the database information is insufficient to answer the claim

Make sure to return the result you choose in the following format:
<answer>
{answer}
</answer>

where the `answer` should be one of `TRUE`, `FALSE`, or `HEDGE` only.
"""

SEARCHER_USER_PROMPT = """
Now, begin to fact check this claim:
<claim>
{claim}
</claim>

The retrieved information from the database is as follows:
<retrieved_information>
{retrieved_information}
</retrieved_information>

Output your fact-checking result (within <answer>...</answer> tags):
"""
