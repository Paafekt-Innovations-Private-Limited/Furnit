# Android agent rules

In addition to the repository-wide `../AGENTS.md`, use this timing rule for
documentation that tracks Android code:

- Do not update code-linked documentation after each implementation, review, test,
  or intermediate correction.
- When the user requests a Git push (including requests such as `push all`), first
  inspect the complete outgoing code diff and then update the nearest owning Android
  documentation and any affected index once, immediately before the commit/push.
- Documentation at that push boundary must describe the final verified code being
  pushed, not intermediate states from the working session.
- If no push is requested, leave code-linked documentation unchanged unless the user
  explicitly asks for documentation work.
- Continue recording only confirmed deferred or unused findings; at the push boundary,
  place them in `docs/DEFERRED.md` when they need to be preserved.
