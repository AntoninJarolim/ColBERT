#!/usr/bin/zsh


HOST="pcknot6"


if ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" exit 2>/dev/null; then
    echo "SSH connection successful (no password needed)."
else
    echo "SSH connection failed or requires a password." >&2
    ssh-add ~/.ssh/fado-generic-karolina
fi

rsync_command='rsync -r experiments/ pcknot6:/mnt/minerva1/nlp/projects/FIR_LLM/experiments'
timeout 2d watchfiles $rsync_command  ./experiments

