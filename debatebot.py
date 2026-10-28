import os
from openai import OpenAI


# NOTE: put your real key here (or better, load it from an environment
# variable) - never leave this checked into source control with a real key in it.
client = OpenAI(
    api_key="",
    base_url="https://sammytymax-2590-resource.services.ai.azure.com/openai/v1",
)

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


def ai_response(conversation_history,topic=None, position=None, level=None):
    """
    Sends the FULL conversation so far to the model so the model has memory of
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

    response = client.responses.create(
    model="gpt-6-sol",  # Replace with your deployed model ID (e.g., gpt-6-sol or gpt-6.1-sol)
    input=conversation_history,
    instructions=system_instruction,
    max_output_tokens=16384,
    reasoning={
        "effort": "low"  # Options: low, medium, max (gpt-6.1-sol defaults to medium)
    }
    )
    return response.text


def get_feedback(conversation_history, topic=None, position=None):
    """
    Takes the full finished transcript and asks the model to act as a coach
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

    response = client.responses.create(
        model="gpt-6-sol",  # Replace with your deployed model ID (e.g., gpt-6-sol or gpt-6.1-sol)
        input=prompt,
        max_output_tokens=16384,
        reasoning={
            "effort": "low"  # Options: low, medium, max (gpt-6.1-sol defaults to medium)
        }
    )
    return response.text
