#!/bin/sh
# check_version.sh TAG
#
# Passes when fasttree.h's FASTTREE_VERSION_MAJOR, _MINOR and _PATCH spell
# its FASTTREE_VERSION_STRING, and TAG is v<that string>-miint, the form of
# this fork's release tags.
set -u
tag=$1
get() { sed -n "s/^#define FASTTREE_VERSION_$1[[:space:]]*//p" fasttree.h; }
string=$(get STRING | tr -d '"')
numbers=$(get MAJOR).$(get MINOR).$(get PATCH)
if [ "$numbers" != "$string" ]; then
  echo "fasttree.h: version numbers $numbers do not match FASTTREE_VERSION_STRING \"$string\""
  exit 1
fi
if [ "$tag" != "v$string-miint" ]; then
  echo "tag $tag does not name fasttree.h's version $string (expected v$string-miint)"
  exit 1
fi
echo "tag $tag matches fasttree.h's version $string"
