from google import genai
from google.genai import types

# NOTE: put your real key here (or better, load it from an environment
# variable) - never leave this checked into source control with a real key in it.
client = genai.Client(api_key="AQ.Ab8RN6KNNyr-FrxUxTwAEC3eXxYvxnYsonYPxPMbdOQScOJGvQ")

DEBATE_SYSTEM_INSTRUCTIONS = (
    "You are a sharp debate challenger. You always take up the position "
    "OPPOSITE the user's stated position on the topic. Challenge the user's "
    "points directly, ask for evidence, and point out weaknesses in their "
    "reasoning. Keep responses focused and no longer than a few sentences "
    "so the debate stays snappy."
)

FEEDBACK_SYSTEM_INSTRUCTIONS = (
    "You are a supportive but honest debate coach reviewing a finished debate "
    "transcript between a user and an AI opponent. You are not debating - you "
    "are analyzing how the user argued. "
    "Give feedback as a short scorecard covering: Evidence Use, Responsiveness "
    "(did they engage with the opponent's points or dodge them), Logical "
    "Consistency, and Persuasiveness. For each, give a rating (Weak/Solid/Strong) "
    "plus one concrete example quoted or paraphrased from the user's own turns. "
    "Then include a section called 'Weakest moment' that identifies the user's "
    "single weakest line and rewrites it as a stronger version. Finish with a "
    "'Watch out for next time' section predicting the counter-argument the user "
    "is most vulnerable to. Keep the whole thing tight and skimmable."
)


def _build_contents(conversation_history):
    """
    conversation_history is a list of turns like:
        [{"role": "user", "message": "..."}, {"role": "ai", "message": "..."}]
    Converts it into the multi-turn `contents` format the Gemini API expects,
    so each new call includes the full running conversation, not just the
    latest message.
    """
    contents = []
    for turn in conversation_history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=turn["message"])])
        )
    return contents


def ai_response(conversation_history, topic=None, position=None, level=None):
    """
    Sends the FULL conversation so far to Gemini so the model has memory of
    earlier turns, rather than only seeing the latest message.
    conversation_history should already include the user's newest message
    as the last entry.
    """
    system_instruction = DEBATE_SYSTEM_INSTRUCTIONS
    if topic or position or level:
        system_instruction += (
            f"\n\nDebate context - Topic: {topic}. The user's position is: "
            f"{position}. Intensity level: {level}."
        )

    contents = _build_contents(conversation_history)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            max_output_tokens=1500,
            system_instruction=system_instruction,
        ),
        contents=contents,
    )
    return response.text


def get_feedback(conversation_history, topic=None, position=None):
    """
    Takes the full finished transcript and asks Gemini to act as a coach
    rather than an opponent, producing a scorecard-style critique instead
    of another rebuttal.
    """
    transcript_lines = []
    for turn in conversation_history:
        speaker = "User" if turn["role"] == "user" else "AI Opponent"
        transcript_lines.append(f"{speaker}: {turn['message']}")
    transcript_text = "\n".join(transcript_lines)

    context = ""
    if topic or position:
        context = f"Topic: {topic}\nUser's position: {position}\n\n"

    prompt = f"{context}Here is the full debate transcript:\n\n{transcript_text}"

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            max_output_tokens=1500,
            system_instruction=FEEDBACK_SYSTEM_INSTRUCTIONS,
        ),
        contents=[prompt],
    )
    return response.text