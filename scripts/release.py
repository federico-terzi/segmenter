#!/usr/bin/env python3

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "target" / "release-artifacts"
BINARY_NAME = "segmenter"
WINDOWS_BINARY_NAME = "segmenter.exe"
EXPECTED_TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
)


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def log(message):
    print(f"INFO: {message}")


def run(args, *, capture=False, check=True):
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and result.returncode != 0:
        command = " ".join(args)
        if capture and result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        fail(f"command failed: {command}")
    return result


def require_tools():
    for tool in ("cargo", "gh", "git", "rustup"):
        if shutil.which(tool) is None:
            fail(f"required tool is not on PATH: {tool}")
    run(["gh", "auth", "status"])


def package_version():
    manifest = REPO_ROOT / "Cargo.toml"
    if tomllib is not None:
        with manifest.open("rb") as file:
            cargo_toml = tomllib.load(file)
        return cargo_toml["package"]["version"]

    in_package = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[package]":
            in_package = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_package = False
        if in_package and stripped.startswith("version"):
            _key, value = stripped.split("=", 1)
            return value.strip().strip('"')

    fail("could not read package.version from Cargo.toml")


def ensure_tracked_worktree_clean():
    result = run(["git", "status", "--porcelain", "--untracked-files=no"], capture=True)
    if result.stdout.strip():
        fail("tracked files are dirty; commit or stash changes before releasing")


def current_os_targets():
    system = platform.system()
    if system == "Darwin":
        return ("aarch64-apple-darwin", "x86_64-apple-darwin")
    if system == "Windows":
        return ("x86_64-pc-windows-msvc",)
    fail(f"unsupported release platform: {system}")


def executable_name(target):
    if target == "x86_64-pc-windows-msvc":
        return WINDOWS_BINARY_NAME
    return BINARY_NAME


def asset_name(version, target):
    suffix = ".exe" if target == "x86_64-pc-windows-msvc" else ""
    return f"{BINARY_NAME}-v{version}-{target}{suffix}"


def expected_asset_names(version):
    names = []
    for target in EXPECTED_TARGETS:
        binary = asset_name(version, target)
        names.append(binary)
        names.append(f"{binary}.sha256")
    return set(names)


def ensure_targets_installed(targets):
    result = run(["rustup", "target", "list", "--installed"], capture=True)
    installed = set(result.stdout.splitlines())
    missing = [target for target in targets if target not in installed]
    if missing:
        fail("missing Rust targets: " + ", ".join(missing))


def build_targets(targets):
    for target in targets:
        log(f"building {target}")
        run(["cargo", "build", "--release", "--target", target])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_artifacts(version, targets):
    release_dir = ARTIFACT_ROOT / f"v{version}"
    release_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    for target in targets:
        built_binary = (
            REPO_ROOT / "target" / target / "release" / executable_name(target)
        )
        if not built_binary.exists():
            fail(f"built binary was not found: {built_binary}")

        binary_name = asset_name(version, target)
        staged_binary = release_dir / binary_name
        shutil.copy2(built_binary, staged_binary)

        if target != "x86_64-pc-windows-msvc":
            mode = staged_binary.stat().st_mode
            staged_binary.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        digest = sha256(staged_binary)
        checksum = release_dir / f"{binary_name}.sha256"
        checksum.write_text(f"{digest}  {binary_name}\n", encoding="utf-8")
        staged.append((staged_binary, digest))
        staged.append((checksum, sha256(checksum)))

    for path, digest in staged:
        print(f"SHA256 {digest} {path.name}")

    return [path for path, _digest in staged]


def release_view(tag):
    result = run(
        ["gh", "release", "view", tag, "--json", "assets,isDraft,tagName"],
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.lower()
        if "not found" in stderr or "release not found" in stderr:
            return None
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        fail(f"could not inspect release {tag}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        fail(f"could not parse gh release view output: {error}")


def release_assets(release):
    return {asset["name"] for asset in release.get("assets", [])}


def ensure_local_tag(tag):
    head = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    local_tag = run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{}}"],
        capture=True,
        check=False,
    )
    if local_tag.returncode == 0:
        tagged_commit = local_tag.stdout.strip()
        if tagged_commit != head:
            fail(f"local tag {tag} exists but does not point at HEAD")
        log(f"local tag {tag} already points at HEAD")
        return

    log(f"creating local tag {tag}")
    run(["git", "tag", tag])


def ensure_remote_tag(tag):
    remote_tag = run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
        capture=True,
        check=False,
    )
    if remote_tag.returncode != 0:
        fail("could not inspect remote tags on origin")
    if remote_tag.stdout.strip():
        log(f"remote tag {tag} already exists")
        return

    log(f"pushing tag {tag}")
    run(["git", "push", "origin", tag])


def create_draft_release(tag, version):
    ensure_local_tag(tag)
    ensure_remote_tag(tag)
    log(f"creating draft release {tag}")
    run(
        [
            "gh",
            "release",
            "create",
            tag,
            "--draft",
            "--title",
            f"segmenter v{version}",
            "--notes",
            f"Segmenter v{version}",
        ]
    )


def ensure_release_ready_for_upload(tag, version):
    release = release_view(tag)
    if release is None:
        create_draft_release(tag, version)
        release = release_view(tag)
        if release is None:
            fail(f"release {tag} was created but cannot be read")
        return release

    if not release.get("isDraft", False):
        existing = release_assets(release)
        missing = expected_asset_names(version) - existing
        if missing:
            fail(
                f"release {tag} is already published and missing assets: "
                + ", ".join(sorted(missing))
            )
        fail(f"release {tag} is already published")

    log(f"using existing draft release {tag}")
    return release


def upload_assets(tag, release, staged_paths):
    existing = release_assets(release)
    duplicates = [path.name for path in staged_paths if path.name in existing]
    if duplicates:
        fail("release already has assets: " + ", ".join(duplicates))

    log(f"uploading {len(staged_paths)} assets to {tag}")
    run(["gh", "release", "upload", tag, *[str(path) for path in staged_paths]])


def publish_if_complete(tag, version):
    release = release_view(tag)
    if release is None:
        fail(f"release {tag} disappeared after upload")

    existing = release_assets(release)
    missing = expected_asset_names(version) - existing
    if missing:
        log("release remains draft; missing assets: " + ", ".join(sorted(missing)))
        return

    if release.get("isDraft", False):
        log(f"all expected assets are present; publishing {tag}")
        run(["gh", "release", "edit", tag, "--draft=false"])
    else:
        log(f"release {tag} is already published")


def main():
    os.chdir(REPO_ROOT)
    require_tools()
    ensure_tracked_worktree_clean()

    version = package_version()
    tag = f"v{version}"
    targets = current_os_targets()

    log(f"preparing release {tag}")
    ensure_targets_installed(targets)
    build_targets(targets)
    staged_paths = stage_artifacts(version, targets)

    release = ensure_release_ready_for_upload(tag, version)
    upload_assets(tag, release, staged_paths)
    publish_if_complete(tag, version)


if __name__ == "__main__":
    main()
