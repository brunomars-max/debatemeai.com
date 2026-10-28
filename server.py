import flask
import debatebot

# In-memory conversation store (no database yet).
# Structure: { userid: {"topic": ..., "position": ..., "level": ...,
#                        "turns": [ {"role": "user"/"ai", "message": "..."} ] } }
# NOTE: this resets whenever the server restarts, and only works with a
# single Flask process - fine for now, but it's a stopgap, not a real store.
conversations = {}

app = flask.Flask('DebateMeAi')


@app.route('/confirmarea', methods=['POST', 'GET'])
def confirmarea():
    if flask.request.method == "POST":
        print("POSTED")
        print(flask.request.form)
    return flask.render_template('confirmarea.html', info=flask.request.form)


@app.route('/debatearea', methods=["POST", "GET"])
def debatearea():
    if flask.request.method == "POST":
        print("SOMEONE HAS JUST SENT A MESSAGE")
        print(flask.request.form)

        inputmessage = flask.request.form["InputMessage"]
        topic = flask.request.form["topic"]
        position = flask.request.form["position"]
        level = flask.request.form["level"]
        userid = flask.request.form["userid"]

        # Start a new conversation record for this user if we haven't seen them yet.
        if userid not in conversations:
            conversations[userid] = {
                "topic": topic,
                "position": position,
                "level": level,
                "turns": [],
            }

        convo = conversations[userid]

        # Add the user's new message to the running transcript.
        convo["turns"].append({"role": "user", "message": inputmessage})

        # Send the WHOLE conversation so far to the model, not just this message.
        answer = debatebot.ai_response(
            convo["turns"],
            topic=convo["topic"],
            position=convo["position"],
            level=convo["level"],
        )

        # Record the AI's reply too, so the next turn has full context.
        convo["turns"].append({"role": "assistant", "message": answer})

        # Return the full updated conversation so the client can just overwrite
        # its localStorage copy instead of the two drifting out of sync.
        return flask.jsonify({"answer": answer, "conversation": convo["turns"]})

    return flask.render_template('debatearea.html')


@app.route('/feedback', methods=["POST"])
def feedback():
    """
    Called when the user ends a debate. Sends the full transcript to a
    separate 'coach' persona (see debatebot.get_feedback) instead of
    continuing the debate itself.
    """
    userid = flask.request.form["userid"]
    convo = conversations.get(userid)

    if not convo or not convo["turns"]:
        return flask.jsonify({"feedback": "No debate found for this session yet."})

    feedback_text = debatebot.get_feedback(
        convo["turns"],
        topic=convo["topic"],
        position=convo["position"],
    )
    return flask.jsonify({"feedback": feedback_text})


@app.route('/topicarea')
def topicarea():
    return flask.render_template('topicarea.html')


@app.route('/')
def homepage():
    return flask.render_template('homepage.html')


@app.route('/contactpage')
def contactpage():
    return flask.render_template('contactpage.html')


@app.route('/how it works')
def howitworks():
    return flask.render_template('howitworks.html')


@app.route('/pricingpage')
def pricingpage():
    return flask.render_template('pricingpage.html')


if __name__ == '__main__':
    app.run()
