#!/usr/bin/env bash
# fetch_one_replica.sh <day> <variant>: one day from a named replica, as reingest_v2.sh's fetch (section 55)
set -u; d=$1; v=$2; out=~/Data/adsb_raw${d:0:4}/$d; repo=adsblol/globe_history_${d:0:4}; tag="v${d//-/.}-planes-readsb-$v"
export GH_TOKEN=$(gh auth token)
rm -rf $out; mkdir -p $out; cd $out || exit 1
gh api repos/$repo/releases/tags/$tag --jq '.assets[] | select(.name | startswith("'$tag'.tar")) | "\(.id) \(.name) \(.size)"' > assets.txt
while read -r id name size; do
  for try in 1 2 3 4 5 6 7 8 9 10; do
    curl -sfL --http1.1 -C - --retry 3 -o $name -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/octet-stream" \
      https://api.github.com/repos/$repo/releases/assets/$id && break; sleep 10; done
  [ "$(stat -c %s $name)" = "$size" ] || { echo "$d $v: size mismatch on $name"; exit 1; }
done < assets.txt
cat $tag.tar.a? > $tag.tar && rm -f $tag.tar.a?
tar -xf $tag.tar || { echo "$d $v: tar reported errors"; exit 1; }
rm -f $tag.tar assets.txt; touch .complete; rm -f ~/Data/adsb_raw${d:0:4}/$d.fetch_failed
echo "$d $v: fetched $(du -sh . | cut -f1), $(find traces -type f | wc -l) trace files"
