#!/usr/bin/env bash
post=$1
grep -onf patterns.txt $post
