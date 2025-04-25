# Ubuntu settings
# export PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$"
export PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\[\033[00m\]\$"

# UV Initialization
cd synthetic-data-generators
curl -LsSf https://astral.sh/uv/install.sh | sh
uv self update
uv --version
uv python install=3.13 --link-mode=copy
uv init --python=3.13 --link-mode=copy --lib --name="synthetic-data-generators" --description="Helper files/functions/classes for generic Python processes"
uv add --python=3.13 --link-mode=copy --no-cache typeguard
uv add --python=3.13 --link-mode=copy --no-cache --requirements=requirements/root.txt
uv add --python=3.13 --link-mode=copy --no-cache --requirements=requirements/dev.txt --group=dev
uv add --python=3.13 --link-mode=copy --no-cache --requirements=requirements/docs.txt --group=docs
uv add --python=3.13 --link-mode=copy --no-cache --requirements=requirements/test.txt --group=test
uv lock --python=3.13 --link-mode=copy --no-cache
uv sync --python=3.13 --link-mode=copy --no-cache --all-groups
uv run pre-commit install
uv run pre-commit autoupdate

# UV Synchronization
uv sync --python=3.13 --link-mode=copy --no-cache --all-groups
uv run --link-mode=copy pre-commit install
uv run --link-mode=copy pre-commit autoupdate
