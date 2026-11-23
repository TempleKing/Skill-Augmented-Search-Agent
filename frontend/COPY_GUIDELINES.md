# TRACE interface writing

Write as a capable research collaborator: precise about the task, candid about the evidence, and clear about the next action. The tone draws on the direct, conversational language of contemporary developer tools; it does not reproduce an official OpenAI interface or imply an affiliation.

## Voice

- Use fluent, idiomatic English. Prefer a precise verb and a concrete object to an abstract slogan.
- Address the user naturally. Use contractions where they make a message easier to read.
- Distinguish what the agent has done from what the user can do next. Avoid claims that go beyond the returned evidence or service state.
- Keep control labels brief. Use complete sentences for explanations, limitations, and recovery instructions.
- Use sentence case for labels and headings. Reserve uppercase for short structural metadata.
- Treat terminology consistently: a question is the user’s input, a run is one execution, a source is an original document, and an excerpt is a passage from that source.

## Examples

| Context | Preferred copy |
| --- | --- |
| Initial prompt | What would you like to investigate? |
| Input guidance | Describe the question, the clues you have, and what you need to establish. |
| Primary action | Start research |
| Running state | Working through your question |
| Empty evidence panel | The evidence will appear here |
| Result guidance | Review the source excerpts and activity to assess how this answer was established. |
| Missing evidence | The retrieved evidence does not fully support an answer. Review the sources or refine the question. |
| Connection failure | Couldn’t reach the backend. Check that the service is running, then try again. |

## Accuracy and grammar

Do not suggest that configuration checks verify provider credentials or that an in-flight request exposes live agent progress. Label historical evaluation figures with their actual denominator and scope. Preserve original questions, source passages, answers, and diagnostic records rather than rewriting them as interface copy.

Handle singular and plural forms for counts. Use “API tokens” for the usage metric, and describe the configured token threshold as a threshold rather than a guaranteed hard limit. Recovery messages should identify the issue and give a relevant next step without blaming the user.
