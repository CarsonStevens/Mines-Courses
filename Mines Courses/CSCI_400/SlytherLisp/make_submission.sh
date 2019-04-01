#!/usr/bin/env bash
if ! [[ -f README.rst ]]; then
    echo "This script MUST be run from the same directory as README.rst" 1>&2
    exit 1
fi
mkdir -p ~/.cache
tar --exclude-vcs-ignores --exclude-vcs -jcf ~/.cache/submission.tar.bz2 .
mv ~/.cache/submission.tar.bz2 .
echo "submission.tar.bz2 created"
