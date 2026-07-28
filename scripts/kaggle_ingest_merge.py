#!/usr/bin/env python3
"""Run Kaggle skill ingest into a temp DB, then merge into the project DB.

The NVIDIA Kaggle skill owns the API clients. This wrapper keeps the skill
unchanged but avoids direct writes to our long-lived research databases.
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path("/home/will/projects/ai-agent-security-2026")
SKILL_ROOT = Path("/home/will/.agents/skills/nvidia-kaggle-skill")


def _load_project_env(root: Path, env: dict[str, str]) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in env:
            continue
        env[key] = value.strip().strip('"').strip("'")


def _run_skill(
    *,
    root: Path,
    skill_root: Path,
    temp_root: Path,
    python: Path,
    args: list[str],
) -> None:
    env = os.environ.copy()
    _load_project_env(root, env)
    env["PROJECT_ROOT"] = str(temp_root)
    env.setdefault("UV_CACHE_DIR", "/tmp/jed-kaggle-uv-cache")
    subprocess.run(
        [str(python), *args],
        cwd=skill_root,
        env=env,
        check=True,
    )


def _copy_if_missing(source: Path, target: Path) -> bool:
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _merge_competition_info(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO competition_info
            (competition_id, title, description, evaluation_metric, url, deadline, updated_at)
        SELECT competition_id, title, description, evaluation_metric, url, deadline, updated_at
        FROM src.competition_info
        WHERE true
        ON CONFLICT(competition_id) DO UPDATE SET
            title = excluded.title,
            description = excluded.description,
            evaluation_metric = excluded.evaluation_metric,
            url = excluded.url,
            deadline = excluded.deadline,
            updated_at = excluded.updated_at
        """
    )


def _merge_discussions(source: Path, target: Path) -> dict[str, int | bool]:
    if not source.exists():
        raise FileNotFoundError(source)
    if _copy_if_missing(source, target):
        with sqlite3.connect(target) as copied:
            discussions = copied.execute("SELECT COUNT(*) FROM discussions").fetchone()[0]
            comments = copied.execute("SELECT COUNT(*) FROM discussion_comments").fetchone()[0]
        return {"copied": True, "discussions": discussions, "comments": comments}

    before_discussions = _count(target, "discussions")
    before_comments = _count(target, "discussion_comments")
    with sqlite3.connect(target) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("ATTACH DATABASE ? AS src", (str(source),))
        conn.execute(
            """
            INSERT INTO discussions
                (competition_id, discussion_id, title, author, author_username, author_tier,
                 votes, comment_count, body_markdown, url, tags,
                 created_at, updated_at, last_fetched_at, ingested_at)
            SELECT competition_id, discussion_id, title, author, author_username, author_tier,
                   votes, comment_count, body_markdown, url, tags,
                   created_at, updated_at, last_fetched_at, ingested_at
            FROM src.discussions
            WHERE true
            ON CONFLICT(competition_id, discussion_id) DO UPDATE SET
                title = excluded.title,
                author = CASE
                    WHEN excluded.author != '' THEN excluded.author
                    ELSE discussions.author
                END,
                author_username = CASE
                    WHEN excluded.author_username != '' THEN excluded.author_username
                    ELSE discussions.author_username
                END,
                author_tier = CASE
                    WHEN excluded.author_tier != '' THEN excluded.author_tier
                    ELSE discussions.author_tier
                END,
                votes = excluded.votes,
                comment_count = CASE
                    WHEN excluded.comment_count > discussions.comment_count THEN excluded.comment_count
                    ELSE discussions.comment_count
                END,
                body_markdown = CASE
                    WHEN length(excluded.body_markdown) > length(discussions.body_markdown)
                    THEN excluded.body_markdown
                    ELSE discussions.body_markdown
                END,
                url = CASE
                    WHEN excluded.url != '' THEN excluded.url
                    ELSE discussions.url
                END,
                tags = CASE
                    WHEN excluded.tags != '[]' THEN excluded.tags
                    ELSE discussions.tags
                END,
                created_at = COALESCE(discussions.created_at, excluded.created_at),
                updated_at = COALESCE(excluded.updated_at, discussions.updated_at),
                last_fetched_at = COALESCE(excluded.last_fetched_at, discussions.last_fetched_at),
                ingested_at = excluded.ingested_at
            """
        )
        conn.execute(
            """
            INSERT INTO discussion_comments
                (discussion_id, competition_id, author, author_username, author_tier,
                 votes, body_markdown, created_at)
            SELECT s.discussion_id, s.competition_id, s.author, s.author_username,
                   s.author_tier, s.votes, s.body_markdown, s.created_at
            FROM src.discussion_comments s
            WHERE NOT EXISTS (
                SELECT 1
                FROM discussion_comments c
                WHERE c.competition_id = s.competition_id
                  AND c.discussion_id = s.discussion_id
                  AND c.author = s.author
                  AND COALESCE(c.created_at, '') = COALESCE(s.created_at, '')
                  AND c.body_markdown = s.body_markdown
            )
            """
        )
        _sync_discussion_comment_counts(conn)
        _merge_competition_info(conn)
        conn.commit()
        conn.execute("DETACH DATABASE src")

    after_discussions = _count(target, "discussions")
    after_comments = _count(target, "discussion_comments")
    return {
        "copied": False,
        "discussions": after_discussions,
        "comments": after_comments,
        "new_discussions": after_discussions - before_discussions,
        "new_comments": after_comments - before_comments,
    }


def _sync_discussion_comment_counts(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE discussions
        SET comment_count = CASE
            WHEN (
                SELECT COUNT(*)
                FROM discussion_comments c
                WHERE c.competition_id = discussions.competition_id
                  AND c.discussion_id = discussions.discussion_id
            ) > comment_count
            THEN (
                SELECT COUNT(*)
                FROM discussion_comments c
                WHERE c.competition_id = discussions.competition_id
                  AND c.discussion_id = discussions.discussion_id
            )
            ELSE comment_count
        END
        """
    )


def _merge_kernels(source: Path, target: Path) -> dict[str, int | bool]:
    if not source.exists():
        raise FileNotFoundError(source)
    if _copy_if_missing(source, target):
        return {"copied": True, "kernels": _count(target, "kernels")}

    before = _count(target, "kernels")
    with sqlite3.connect(target) as conn:
        conn.execute("ATTACH DATABASE ? AS src", (str(source),))
        conn.execute(
            """
            INSERT INTO kernels
                (competition_id, kernel_ref, title, author,
                 total_votes, last_run_time, is_private, ingested_at)
            SELECT competition_id, kernel_ref, title, author,
                   total_votes, last_run_time, is_private, ingested_at
            FROM src.kernels
            WHERE true
            ON CONFLICT(competition_id, kernel_ref) DO UPDATE SET
                title = excluded.title,
                author = excluded.author,
                total_votes = excluded.total_votes,
                last_run_time = excluded.last_run_time,
                is_private = excluded.is_private,
                ingested_at = excluded.ingested_at
            """
        )
        _merge_competition_info(conn)
        conn.commit()
        conn.execute("DETACH DATABASE src")
    after = _count(target, "kernels")
    return {"copied": False, "kernels": after, "new_kernels": after - before}


def _count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _discussion(args: argparse.Namespace) -> dict[str, int | bool]:
    root = Path(args.root)
    skill_root = Path(args.skill_root)
    python = Path(args.python)
    with tempfile.TemporaryDirectory(prefix="jed-kaggle-discussions-") as temp:
        temp_root = Path(temp)
        command = [
            "scripts/discussion_ingest.py",
            args.competition_id,
            "--max-pages",
            str(args.max_pages),
            "--sort-by",
            args.sort_by,
            "--page-size",
            str(args.page_size),
        ]
        if not args.fetch_comments:
            command.append("--nofetch-comments")
        _run_skill(
            root=root,
            skill_root=skill_root,
            temp_root=temp_root,
            python=python,
            args=command,
        )
        return _merge_discussions(
            temp_root / "data" / "discussions.db",
            root / "data" / "discussions.db",
        )


def _kernel(args: argparse.Namespace) -> dict[str, int | bool]:
    root = Path(args.root)
    skill_root = Path(args.skill_root)
    python = Path(args.python)
    with tempfile.TemporaryDirectory(prefix="jed-kaggle-kernels-") as temp:
        temp_root = Path(temp)
        command = [
            "scripts/kernel_ingest.py",
            args.competition_id,
            "--max-pages",
            str(args.max_pages),
            "--sort-by",
            args.sort_by,
            "--page-size",
            str(args.page_size),
        ]
        _run_skill(
            root=root,
            skill_root=skill_root,
            temp_root=temp_root,
            python=python,
            args=command,
        )
        return _merge_kernels(
            temp_root / "data" / "kernels.db",
            root / "data" / "kernels.db",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--skill-root", default=str(SKILL_ROOT))
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    subparsers = parser.add_subparsers(dest="kind", required=True)

    discussion = subparsers.add_parser("discussion")
    discussion.add_argument("competition_id")
    discussion.add_argument("--max-pages", type=int, default=2)
    discussion.add_argument("--sort-by", default="updated")
    discussion.add_argument("--page-size", type=int, default=20)
    discussion.add_argument("--fetch-comments", action="store_true")

    kernel = subparsers.add_parser("kernel")
    kernel.add_argument("competition_id")
    kernel.add_argument("--max-pages", type=int, default=3)
    kernel.add_argument("--sort-by", default="dateRun")
    kernel.add_argument("--page-size", type=int, default=20)

    args = parser.parse_args()
    result = _discussion(args) if args.kind == "discussion" else _kernel(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
