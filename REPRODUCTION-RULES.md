# Rules for reproduction reports

These are my rules, not a standard. The wording of the categories follows ACM Artifact Review and Badging v1.1 (2020).

1. **Reproduced** means someone outside the author's team ran the published code on the pinned data and reports numbers. **Replicated** means they wrote their own implementation. The author re-running the code is a repeat, never a reproduction.
2. A report gives the commit hash, model revisions, hardware, operating system, library versions, the exact commands, the raw output and any deviation. A report without raw output cannot be counted.
3. Agreement is judged against `expected/tolerances.json`, which was set before the first outside run and changes only with a dated note. Inside tolerance is "reproduced". Outside is "not reproduced" or "partly reproduced", and is published as such.
4. I may write "reproduced by N outside parties, reports linked" with the count. I may not write "independently validated", "peer reviewed" or "verified". I may not count my own runs, or runs by collaborators without saying so, or a run on different data.
5. Failed and divergent reports are listed in `REPRODUCTIONS.md` with the same prominence as successes. I may reply in the issue. I may not delete or edit a report. If a report finds a bug, I fix it in a new tagged release, mark the old claim superseded and credit the reporter.
6. Reporters are credited by name or handle and date. They are added to the archived metadata only with their consent.
7. A headline number on the websites states how many outside reproductions back it, including none.
