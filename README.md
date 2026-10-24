# wikilogreader

Data collection for a small model that predicts which English Wikipedia edits
will be reverted. This is phase 1: a collector that records edits as they
happen, fetches each edit's diff, and labels it 48 hours later with whether it
was reverted and how quickly.

Later phases (baselines, a teacher model, a distilled student that runs in the
browser) will live in this repo too.

## What it collects

For every edit to an article on English Wikipedia (`wiki = enwiki`,
`type = edit`, namespace 0):

| Table | Contents | When |
|---|---|---|
| `edits` | revision ids, page id, title, username, bot flag, anonymous or not, temporary account or not, comment, minor flag, old and new size, timestamp | as the edit arrives on the stream |
| `edits` (enrichment columns) | account registration date, edit count, user groups, change tags on the revision | about a minute later, 50 edits per request |
| `diffs` | added and removed text, the full old and new versions of changed lines, line counts, and the original diff HTML (compressed) | as soon as possible after the edit |
| `labels` | tags at check time, reverted yes/no, the reverting revision, revert time in seconds, how it was reverted, whether it was a self-revert | 48 hours after the edit |

"Anonymous" covers two cases. Since 4 November 2025 English Wikipedia gives
logged-out editors a temporary account named like `~2025-12345-67` instead of
showing their IP address, so nearly all anonymous edits now look like that.
Older-style IP usernames are still recognised.

### How edits are labelled

1. 48 hours after an edit, the collector reads the revision's change tags.
   MediaWiki adds `mw-reverted` when a later edit undoes it by undo, rollback,
   or a manual edit that restores an earlier version exactly.
2. For reverted edits it reads the page history from the edit's parent onwards
   and takes the first later revision tagged `mw-rollback`, `mw-undo` or
   `mw-manual-revert`, or that restores the parent's exact content (same
   sha1). That gives the reverting revision and the time to revert.
3. Revisions that no longer exist (deleted page, removed revision) are stored
   with `status = 'missing'` and no label.

`reverted_48h` is the label to train on: reverted, and the revert happened
within the 48-hour window.

## Limits of the labels

These matter for how the results can be read later.

- A revert is not the same as vandalism. Good-faith edits get reverted in
  content disputes, for missing sources, or by mistake. The model will learn
  "likely to be reverted", and that is how the demo should describe it.
- Only reverts that MediaWiki recognises count. If someone removes a bad edit
  by hand while also changing something else, that's not an exact revert, so
  the edit is labelled not reverted.
- `mw-reverted` is applied by a background job and only when the reverting
  edit qualifies (see [Manual:Reverts](https://www.mediawiki.org/wiki/Manual:Reverts)).
  Missed tags make some reverted edits look clean.
- A large share of fast reverts come from one bot (ClueBot NG). A model trained
  on these labels partly learns to imitate that bot.
- Self-reverts (someone undoing their own edit) are recorded in
  `reverter_is_self` so they can be excluded.

## Leakage to watch for in phase 2

- **Enrichment happens after the edit.** Edit count and groups are read about a
  minute later, so a fast-moving editor's count includes a few later edits.
  Tags are read then too; `mw-reverted` is removed from `tags_at_save`, but an
  edit that was itself a revert keeps its revert tags (that's a fair feature).
- **The `labels.tags` column is read 48 hours later** and must never be used as
  a feature.
- **Diffs fetched late can fail** because the revision was hidden or the page
  deleted, which happens more to bad edits. Failed diffs (`api_error:*`
  statuses) correlate with the label and should be handled carefully rather
  than dropped silently.
- **Same page, both splits.** Edit wars put several related edits on one page;
  the phase 2 split keeps each page on one side.

## Running it

It needs a machine that stays on for the weeks it runs. There are three ways
to set it up.

### On a Proxmox VE host (what I use)

From the Proxmox web UI, open the host's **Shell** and run:

```sh
bash -c "$(curl -fsSL https://raw.githubusercontent.com/rockminelaw/wikilogreader/main/deploy/proxmox-create-ct.sh)"
```

This creates an unprivileged Debian 13 container (2 cores, 1 GB of memory,
a 64 GB disk, DHCP on `vmbr0`) that starts with the host, and installs the
collector inside it as a systemd service running under its own user. A timer
checks every 10 minutes and restarts the service if no stream events have
been stored for 15 minutes. Settings can be changed with variables, for
example `STORAGE=local-zfs DISK_GB=100`.

Inside the container (`pct enter <id>` on the host):

```sh
journalctl -u wikilogreader -f        # live log, progress line every 5 minutes
wikilogreader stats                   # counts as JSON
wikilogreader backup /var/lib/wikilogreader/edits-copy.sqlite
bash /opt/wikilogreader/deploy/install.sh   # update to the latest code
```

`backup` makes a consistent copy while the collector keeps running. To get the
copy onto the Proxmox host: `pct pull <id> /var/lib/wikilogreader/edits-copy.sqlite /root/edits.sqlite`.
Optional settings go in `/etc/wikilogreader.env` as `WRR_*=value` lines,
followed by `systemctl restart wikilogreader`.

### On any Debian or Ubuntu machine

As root: `git clone https://github.com/rockminelaw/wikilogreader /opt/wikilogreader`
then `bash /opt/wikilogreader/deploy/install.sh`. Same commands as above.

### With Docker

```sh
docker compose up -d --build
docker compose logs -f            # progress line every 5 minutes
docker compose exec collector python -m wikirevert stats
docker compose exec collector python -m wikirevert backup /data/edits-copy.sqlite
docker compose cp collector:/data/edits-copy.sqlite ./edits.sqlite
```

The data lives in the `wrr-data` Docker volume. On Windows or macOS that's
Docker Desktop, and the computer has to stay on and awake.

### Settings

Environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `WRR_DB_PATH` | `data/edits.sqlite` | database file |
| `WRR_LABEL_DELAY_HOURS` | `48` | how old an edit must be before labelling |
| `WRR_REQUESTS_PER_MINUTE` | `150` | ceiling on Action API requests |
| `WRR_MIN_REQUEST_INTERVAL` | `0.25` | seconds between request starts |
| `WRR_MAXLAG` | `5` | `maxlag` sent with each request |
| `WRR_FETCH_BOT_DIFFS` | `1` | set to `0` to skip diffs for bot edits |
| `WRR_API_TOKEN` | empty | personal OAuth 2 access token for a higher rate limit |
| `WRR_STATS_INTERVAL` | `300` | seconds between progress lines |

## How it treats Wikimedia's servers

- User-Agent: `WikiLogReader/0.1.0 (https://github.com/rockminelaw/wikilogreader; celarry1@gmail.com) python-requests/x.y`,
  in the format the
  [User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy)
  asks for.
- One API request at a time, from a single thread, with at most 150 a minute.
  Wikimedia's 2026 [rate limits](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits)
  allow 200 a minute for an unauthenticated client with a proper User-Agent.
  English Wikipedia gets about 1.5 article edits a second, which needs about
  100 requests a minute: one diff per edit plus a batched request for every 50
  edits for enrichment and another for labels.
- `maxlag=5` on every request; on a lag error it waits and retries.
- 429 and 503 responses are retried after the `Retry-After` time (at least 5 s
  without one). Network and server errors back off exponentially with jitter.
  After six failed attempts the worker pauses for a minute.
- The stream reconnects after any drop and sends the stored `Last-Event-ID`,
  so it carries on from where it stopped. EventStreams keeps about a week of
  history, so a longer outage leaves a gap.
- Edits and the stream position are saved in one transaction, so a crash or
  restart neither loses nor duplicates edits.

## Expected size

At about 130,000 article edits a day, a few hundred thousand labelled edits
takes four or five days including the 48-hour wait. In a short test the
stored diff data averaged about 9 KB per edit (median about 2 KB; a few large
list edits pull the average up), so expect roughly 1 GB per 100,000 edits.
That estimate came from only 40 diffs and will be updated once the collector
has run for a while. At that rate the database grows about 1.2 GB a day, so the
Proxmox container's 64 GB disk lasts around seven weeks. Check with `df -h /`
inside the container; on the host, `pct resize <id> rootfs +32G` adds space
without stopping it.

## Privacy

The database holds usernames, including temporary account names, and the text
of edits. That's all public on Wikipedia, but the database isn't committed and
shouldn't be published as is. Nothing derived from it on the website will show
usernames.

## Tests

```sh
pip install '.[test]'
python -m pytest
```

The fixtures in `tests/fixtures` are real responses recorded from
EventStreams and the English Wikipedia API on 24 October 2026.
