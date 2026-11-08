"""Resume the authorized site reconstruction with AnyiOS default models.

Provider cooldowns are honored without substituting models. The image phase can
continue while the text model is cooling down. No build or deployment is run.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


SPEC = importlib.util.spec_from_file_location('rebuild_content', Path(__file__).with_name('rebuild-content.py'))
REBUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REBUILD)
STATE = REBUILD.OUT / 'coordinator-state.json'


def cooldown_until(text: str, timestamp: datetime) -> datetime | None:
    waits = [int(value) for value in re.findall(r'Cooldown wait (\d+)s', text)]
    return timestamp + timedelta(seconds=max(waits)) if waits else None


def saved_cooldown() -> datetime | None:
    deadlines = []
    for line in (REBUILD.OUT / 'events.jsonl').read_text().splitlines():
        entry = json.loads(line)
        deadline = cooldown_until(str(entry.get('error', '')), datetime.fromisoformat(entry['at']))
        if deadline is not None:
            deadlines.append(deadline)
    return max(deadlines) if deadlines else None


def save_state(**fields):
    previous = REBUILD.read_json(STATE) if STATE.exists() else {}
    REBUILD.write_json(STATE, {**previous, **fields, 'pid': os.getpid(),
                              'updated_at': datetime.now(timezone.utc).isoformat()})


async def run_phase(script: str, phase: str | None, args, concurrency: int | None = None) -> tuple[int, str]:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    label = phase or Path(script).stem
    target = REBUILD.OUT / f'coordinator-{stamp}-{label}.log'
    script_path = str(REBUILD.ROOT / 'scripts' / script)
    command = ['node', script_path] if script.endswith('.mjs') else [sys.executable, '-B', script_path]
    if phase:
        command.append(phase)
    if script in {'rebuild-content.py', 'refine-content.py'}:
        command.extend(['--app-root', str(args.app_root)])
    if concurrency is not None:
        command.extend(['--concurrency', str(concurrency)])
    save_state(phase=label, phase_log=str(target.relative_to(REBUILD.ROOT)))
    REBUILD.event('coordinator_phase_started', phase=label, log=str(target))
    started = time.time()
    with target.open('w') as output:
        process = await asyncio.create_subprocess_exec(*command, stdout=output,
                                                       stderr=asyncio.subprocess.STDOUT,
                                                       cwd=REBUILD.ROOT)
        code = await process.wait()
    content = target.read_text()
    # Some stages save provider errors per item and raise a compact batch
    # exception. Include fresh diagnostics when deciding whether to cool down.
    for name in ['visual-failures.json', 'image-repair-failures.json', 'image-failures.json']:
        report = REBUILD.OUT / name
        if report.exists() and report.stat().st_mtime >= started:
            content += '\n' + report.read_text()
    REBUILD.event('coordinator_phase_finished', phase=label, exit_code=code)
    return code, content


def is_audited(page) -> bool:
    folder = REBUILD.page_folder(page)
    path = folder / 'final-audit.json'
    return (REBUILD.polished_article(page) is not None and path.exists()
            and REBUILD.read_json(path).get('body_sha256')
            == REBUILD.digest(REBUILD.read_json(folder / 'reviewed.json')['body_markdown']))


def text_progress(pages) -> tuple[int, int, int]:
    return (sum((REBUILD.page_folder(page) / 'reviewed.json').exists() for page in pages),
            sum(REBUILD.polished_article(page) is not None for page in pages),
            sum(is_audited(page) for page in pages))


def transient_image_failure(output: str) -> bool:
    codes = re.findall(r'"code"\s*:\s*"([a-z_]+)"', output)
    temporary = {'image_capacity_exhausted', 'image_capacity_unavailable', 'image_queue_full',
                 'gateway_concurrency_limit', 'rate_limit_error', 'upstream_error'}
    return (bool(codes) and all(code in temporary for code in codes)
            or bool(re.search(r'HTTP (?:429|500|502|503|504)\b', output)))


def transient_text_failure(output: str) -> bool:
    return bool(re.search(r'HTTP (?:429|500|502|503|504)\b|All connection attempts failed|'
                          r'Connection refused|ConnectTimeout|ReadTimeout|Server disconnected', output, re.I))


async def available_image_capacity() -> int:
    from workflow.service.media_mcp_capacity import image_capacity
    from workflow.utils.image_registry import list_image_profiles, resolve_image_profile

    selected = list_image_profiles()['default_profile_id']
    profile = resolve_image_profile(selected)
    if profile.request_mode != 'chatgpt_web':
        return 4
    return (await image_capacity(profile))['effective_concurrency']


async def run(args):
    sys.path.insert(0, str(args.app_root / 'workflow/src'))
    from workflow.service.media_mcp_capacity import CapacityError

    pages = REBUILD.read_json(REBUILD.OUT / 'inventory.json')['pages']
    if args.skip_visual_review:
        REBUILD.write_json(REBUILD.OUT / 'publication-policy.json', {
            'visual_review': 'skipped_by_user',
            'instruction': '不需要逐图核对标签、箭头和正文关系，也不运行据此自动重画的环节。',
            'updated_at': datetime.now(timezone.utc).isoformat(),
        })
    stop_images = asyncio.Event()
    deadline = saved_cooldown()
    profile_id = REBUILD.default_text_profile().id
    attempts = {}
    save_state(status='resuming', error=None, image_status='starting', image_error=None,
               text_status='running', stopped_at=None, stop_reason=None,
               resumed_at=datetime.now(timezone.utc).isoformat(),
               visual_review='required' if REBUILD.visual_review_required() else 'skipped_by_user')

    async def image_worker():
        if args.initial_image_pid:
            while not stop_images.is_set():
                try:
                    os.kill(args.initial_image_pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(15)
        failures = 0
        while not stop_images.is_set():
            pending = any((article := REBUILD.polished_article(page)) is not None and (
                not REBUILD.diagram_corrections_current(page) or any(
                    REBUILD.current_image_record(REBUILD.page_folder(page), image) is None
                    for image in article['images'])) for page in pages)
            if pending:
                try:
                    capacity = await available_image_capacity()
                except CapacityError as error:
                    # Capacity discovery does not submit a generation. Keep its
                    # bounded service error visible while independent text work
                    # continues; retry on the gateway's next available capacity.
                    save_state(image_status='waiting_for_capacity', image_error=str(error))
                    await asyncio.sleep(60)
                    continue
                if capacity == 0:
                    save_state(image_status='waiting_for_capacity', image_error=None)
                    await asyncio.sleep(60)
                    continue
                save_state(image_status='running', image_error=None)
                code, output = await run_phase('rebuild-content.py', 'images', args,
                                               min(args.image_concurrency, capacity))
                if code and transient_image_failure(output):
                    save_state(image_status='waiting_for_capacity', image_error='Provider temporarily unavailable')
                    await asyncio.sleep(60)
                    continue
                made_progress = bool(re.search(r'^\{.*"event": "image_ready",.*\}$', output, re.M))
                failures = failures + 1 if code and not made_progress else 0
                if failures >= 3:
                    save_state(image_status='needs_attention', image_failure_report='output/content-rebuild/image-failures.json')
                    raise RuntimeError('Image generation failed three times; inspect image-failures.json')
            else:
                await asyncio.sleep(15)

    image_task = asyncio.create_task(image_worker())
    try:
        while not all(is_audited(page) for page in pages):
            # A user-selected new default can be tried immediately; otherwise
            # honor the cooldown reported by this same selected model.
            current_id = REBUILD.default_text_profile().id
            if current_id != profile_id:
                deadline = None
                profile_id = current_id
            now = datetime.now(timezone.utc)
            reviewed, polished, audited = text_progress(pages)
            save_state(total_pages=len(pages), progress={'reviewed': reviewed, 'polished': polished, 'audited': audited})
            if deadline is not None and now < deadline:
                save_state(status='waiting_for_default_text_model', resume_at=deadline.isoformat(),
                           text_model=REBUILD.default_text_profile().model)
                await asyncio.sleep(min(60, (deadline - now).total_seconds()))
                if image_task.done():
                    await image_task
                continue
            save_state(status='running', resume_at=None, error=None)
            if any(not (REBUILD.page_folder(page) / 'reviewed.json').exists() for page in pages):
                script, phase = 'rebuild-content.py', 'draft'
            elif any(REBUILD.polished_article(page) is None for page in pages):
                script, phase = 'rebuild-content.py', 'polish'
            else:
                script, phase = 'refine-content.py', None
            before = text_progress(pages)
            code, output = await run_phase(script, phase, args, args.text_concurrency)
            new_deadline = cooldown_until(output, datetime.now(timezone.utc))
            if new_deadline is not None:
                deadline = new_deadline
                continue
            if transient_text_failure(output):
                # No Retry-After was given. Delay an actual provider retry.
                deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
                continue
            key = phase or script
            after = text_progress(pages)
            attempts[key] = attempts.get(key, 0) + 1 if before == after else 0
            if attempts[key] >= 4:
                raise RuntimeError(f'{key} could not finish after four resumptions; inspect the phase log')
            if image_task.done():
                await image_task
        while any(REBUILD.media_snapshot(page) is None for page in pages):
            save_state(status='waiting_for_images')
            if image_task.done():
                await image_task
                raise RuntimeError('Image generation ended with missing current native images')
            await asyncio.sleep(30)
        stop_images.set()
        await image_task
        code, _ = await run_phase('check-content-formulas.mjs', None, args)
        if code:
            raise RuntimeError('Formula rendering check failed; inspect formula-check.json')
        while REBUILD.visual_review_required() and not all(REBUILD.media_passed(page) for page in pages):
            save_state(status='reviewing_images')
            code, output = await run_phase('rebuild-content.py', 'watch-media', args, args.image_concurrency)
            if code:
                deadline = cooldown_until(output, datetime.now(timezone.utc))
                if deadline is None:
                    raise RuntimeError('Visual review or repair needs attention; inspect the phase log')
                save_state(status='waiting_for_default_text_model', resume_at=deadline.isoformat())
                while datetime.now(timezone.utc) < deadline:
                    await asyncio.sleep(min(60, (deadline - datetime.now(timezone.utc)).total_seconds()))
        for phase in ['integrate', 'navigation']:
            code, _ = await run_phase('rebuild-content.py', phase, args)
            if code:
                raise RuntimeError(phase + ' needs attention; inspect its phase log')
        preview_manifest = REBUILD.SRC / '.vuepress/public/images/home-previews/manifest.json'
        if preview_manifest.is_file():
            code, _ = await run_phase('prepare-home-previews.py', None, args)
            if code:
                raise RuntimeError('Homepage preview refresh failed; inspect its phase log')
        code, _ = await run_phase('validate-rebuilt-content.py', None, args)
        if code:
            raise RuntimeError('Final source validation failed; inspect validation.json')
        reviewed, polished, audited = text_progress(pages)
        save_state(status='completed', completed_at=datetime.now(timezone.utc).isoformat(),
                   progress={'reviewed': reviewed, 'polished': polished, 'audited': audited},
                   text_status='completed', image_status='completed', image_error=None, resume_at=None)
        REBUILD.event('coordinator_completed', pages=len(pages))
    finally:
        stop_images.set()
        if not image_task.done():
            await image_task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-root', required=True, type=Path)
    parser.add_argument('--text-concurrency', type=int, default=6)
    parser.add_argument('--image-concurrency', type=int, default=58)
    parser.add_argument('--initial-image-pid', type=int)
    parser.add_argument('--skip-visual-review', action='store_true',
                        help='Persist the user instruction to integrate native images without visual review/redraw')
    arguments = parser.parse_args()
    with (REBUILD.OUT / '.coordinator.lock').open('a') as coordinator_lock:
        fcntl.flock(coordinator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            asyncio.run(run(arguments))
        except Exception as error:
            save_state(status='needs_attention', error=str(error)[:1000])
            raise
